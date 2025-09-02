#ifndef _ANN_IO_WRITER_HPP
#define _ANN_IO_WRITER_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <string>
#include <ranges>
#include <functional>

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/uio.h>

#include "custom/custom.hpp"
#include "util/intrin.hpp"
#include "util/seq.hpp"
#include "util/helper.hpp"

namespace ANN::io{

class mmap_writer
{
	char *p = nullptr;
	size_t n = 0;

public:
	mmap_writer(const std::string &filename)
	{
		int fd = open(filename.c_str(), O_WRONLY|O_CREAT);
		if(fd==-1)
			throw std::runtime_error("failed to open "+filename);

		struct stat sb;
		if(fstat(fd, &sb)==-1)
			throw std::runtime_error("bad state of "+filename);

		if(!S_ISREG(sb.st_mode))
			throw std::runtime_error(filename+" is not a reg file");

		p = (char*)mmap(NULL, 0/**/, PROT_WRITE, MAP_PRIVATE, fd, 0);
		if(p==MAP_FAILED)
			throw std::runtime_error("failed to map "+filename);

		if(close(fd)==-1)
			throw std::runtime_error("failed to close "+filename);
	}

	~mmap_writer()
	{
		munmap(p,n);
	}
};

class buffered_writer
{
	int fd = -1;

	template<std::ranges::range R>
	static auto calc_offset(const R &r) requires(
		std::is_trivially_copyable_v<std::ranges::range_value_t<R>> &&
		!std::ranges::range<std::ranges::range_value_t<R>>)
	{
		return std::pair(
			util::delayed_seq(std::ranges::size(r)+1, [&](size_t i){
				return i * sizeof(std::ranges::range_value_t<R>);
			}),
			0
		);
	}
	template<std::ranges::range R>
	static auto calc_offset(const R &r) requires
		std::ranges::range<std::ranges::range_value_t<R>>
	{
		using cm = custom<typename lookup_custom_tag<R>::type>;
		using value_t = std::ranges::range_value_t<R>;
		using ret_t = decltype(calc_offset(std::declval<value_t>()));
		auto sub = util::init<typename cm::seq<ret_t>>(
			std::ranges::size(r),
			[&](size_t i){return calc_offset(r[i]);}
		);
		auto sizes = util::delayed_seq(sub.size(), [&](size_t i){
			return sub[i].first.back();
		});
		auto [offset,tot] = cm::scan(sizes);
		offset.push_back(tot);

		return std::pair(std::move(offset), std::move(sub));
	}

	template<std::ranges::range R, class P>
	size_t write_ordered(const R &r, const P &sums, std::byte *out) const
	{
		using namespace std::ranges;
		using cm = custom<typename lookup_custom_tag<R>::type>;
		using value_t = range_value_t<R>;
		constexpr const bool is_trivial = std::is_trivially_copyable_v<value_t>;
		constexpr const bool is_range = range<value_t>;
		static_assert(is_trivial || is_range);
		constexpr const bool is_basecase = !is_range;

		[[maybe_unused]] const auto &[offset, sub] = sums;
		const auto n = size(r);

		auto fill_buffer = [&](std::byte *p, size_t i){
			if constexpr(is_basecase)
			{
				const auto &v = r[i];
				memcpy(p, &v, sizeof(value_t));
			}
			else
				write_ordered(r[i], sub[i], p);
		};

		if(out) // dump data and let the caller write
		{
			cm::parallel_for(0, n, [&](size_t i){
				std::byte *p = out+offset[i];
				fill_buffer(p, i);
			});
			return 0;
		}

		constexpr const size_t buf_limit = 1<<30;
		typename cm::seq<std::byte> buf(buf_limit);
		size_t bytes_written = 0;
		for(size_t i=0; i<n;)
		{
			size_t j = prev(upper_bound(
				offset.begin()+i, offset.end(), offset[i]+buf.size()
			)) - offset.begin();

			if constexpr(is_basecase) // base case cannot go finer
				static_assert(sizeof(value_t)<=buf_limit);
			else if(i==j) // go for finer split
			{
				write_ordered(r[i], sub[i], nullptr);
				i++;
				continue;
			}

			cm::parallel_for(i, j, [&](size_t k){
				std::byte *p = buf.data() + (offset[k]-offset[i]);
				fill_buffer(p, k);
			});
			bytes_written += write(fd, buf.data(), offset[j]-offset[i]);
			i = j;
		}
		return bytes_written;
	}

public:
	buffered_writer(const std::string &filename)
	{
		fd = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC,
			S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		if(fd==-1)
			throw std::runtime_error("failed to open "+filename);

		struct stat sb;
		if(fstat(fd, &sb)==-1)
			throw std::runtime_error("bad state of "+filename);

		if(!S_ISREG(sb.st_mode))
			throw std::runtime_error(filename+" is not a reg file");
	}

	template<typename T>
	size_t operator()(T value) const
		requires(std::is_trivially_copyable_v<T>)
	{
		return write(fd, (void*)&value, sizeof(value));
	}

	template<typename T>
	size_t operator()(T array[], size_t n) const
		requires(std::is_trivially_copyable_v<T>)
	{
		return write(fd, (void*)array, sizeof(T)*n);
	}

	template<std::ranges::range R>
	size_t operator()(const R &r) const
	{
		auto sums = calc_offset(r);
		return write_ordered(r, sums, nullptr);
	}

	~buffered_writer()
	{
		close(fd);
	}
};

} // namespace ANN::io

#endif // _ANN_IO_WRITER_HPP
