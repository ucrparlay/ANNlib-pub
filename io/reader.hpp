#ifndef _ANN_IO_READER_HPP
#define _ANN_IO_READER_HPP

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <cassert>
#include <utility>
#include <string>
#include <initializer_list>
#include <optional>
#include <concepts>
#include <ranges>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "custom/custom.hpp"
#include "util/util.hpp"
#include "util/helper.hpp"
#include "util/seq.hpp"

namespace ANN::io{

class mmap_reader
{
	class transformer_base{
		transformer_base(mmap_reader *r) : r(r), arg(nullptr){}

		transformer_base(const transformer_base&) = delete;
		transformer_base(transformer_base&&) = delete;
		transformer_base& operator=(const transformer_base&) = delete;
		transformer_base& operator=(transformer_base&&) = delete;

		friend mmap_reader;

	protected:
		mmap_reader *r;
		const void *arg;
	};

	template<typename A=void>
	class transformer : transformer_base{
		friend mmap_reader;
	public:
		template<typename T>
		operator T() const{
			if constexpr(std::is_void_v<A>)
				return this->r->read<T>();
			else
				return this->r->read<T>(*(const A*)(this->arg));
		}
	};

	std::byte *base, *cur;
	size_t n;
	transformer_base gen = this;

	template<std::ranges::range R>
	static auto calc_offset(size_t size)
	{
		return std::pair(
			util::delayed_seq(size+1, [&](size_t i){
				return i * sizeof(std::ranges::range_value_t<R>);
			}),
			0
		);
	}
	template<std::ranges::range R, std::ranges::range Seq>
	static auto calc_offset(const Seq &sizes)
	{
		using cm = custom<typename lookup_custom_tag<R>::type>;
		using value_t = std::ranges::range_value_t<R>;
		static_assert(std::ranges::range<value_t>,
			"Dimensions of R and sizes do not match");
		using ret_t = decltype(calc_offset<value_t>(sizes[0]));
		auto sub = util::init<typename cm::seq<ret_t>>(
			std::ranges::size(sizes),
			[&](size_t i){return calc_offset<value_t>(sizes[i]);}
		);
		auto sums = util::delayed_seq(sub.size(), [&](size_t i){
			return sub[i].first.back();
		});
		auto [offset,tot] = cm::scan(sums);
		offset.push_back(tot);

		return std::pair(std::move(offset), std::move(sub));
	}

	template<std::ranges::range R, class P>
	auto read_multidim(const P &sums, std::byte *p) const
	{
		using namespace std::ranges;
		using value_t = range_value_t<R>;
		const auto &[offset, sub] = sums;

		if constexpr(!range<decltype(sub)>)
			return util::init<R>(offset.size()-1, [&](size_t i){
				return *(value_t*)(p+offset[i]);
			});
		else
			return util::init<R>(sub.size(), [&](size_t i){
				return read_multidim<value_t>(sub[i], p+offset[i]);
			});
	}

public:
	mmap_reader(const std::string &filename)
	{
		int fd = open(filename.c_str(), O_RDONLY);
		if(fd==-1)
			throw std::runtime_error("failed to open "+filename);

		struct stat sb;
		if(fstat(fd, &sb)==-1)
			throw std::runtime_error("bad state of "+filename);

		if(!S_ISREG(sb.st_mode))
			throw std::runtime_error(filename+" is not a reg file");
		n = sb.st_size;

		base = (std::byte*)mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
		if(base==MAP_FAILED)
			throw std::runtime_error("failed to map "+filename);
		cur = base;

		if(close(fd)==-1)
			throw std::runtime_error("failed to close "+filename);
	}

	template<typename T>
	[[nodiscard]] T read() requires(std::is_trivially_copyable_v<T>)
	{
		std::byte *p = cur;
		cur += sizeof(T);
		assert(cur <= base+n);
		return *(T*)p;
	}

	template<std::ranges::range R, class U>
	[[nodiscard]] R read(const U &sizes)
	{
		auto sums = calc_offset<R>(sizes);
		std::byte *p = cur;
		cur += sums.first.back();
		assert(cur <= base+n);
		return read_multidim<R>(sums, p);
	}
	template<std::ranges::range R, class U>
	[[nodiscard]] R read(const std::initializer_list<U> &sizes)
	{
		return read<R>(std::ranges::ref_view(sizes));
	}

	[[nodiscard]] transformer<>& operator()()
	{
		gen.arg = nullptr;
		return static_cast<transformer<>&>(gen);
	}

	template<typename U>
	[[nodiscard]] transformer<U>& operator()(const U &sizes)
	{
		gen.arg = &sizes;
		return static_cast<transformer<U>&>(gen);
	}
	template<typename U>
	[[nodiscard]] auto operator()(const std::initializer_list<U> &sizes)
		-> transformer<std::initializer_list<U>>&
	{
		gen.arg = &sizes;
		return static_cast<
			transformer<std::initializer_list<U>>&
			>(gen);
	}

	size_t set_pos(size_t pos)
	{
		assert(pos<=n);
		size_t old = cur-base;
		cur = base + pos;
		return old;
	}

	size_t size() const
	{
		return n;
	}

	~mmap_reader()
	{
		munmap(base,n);
	}
};

} // namespace ANN::io

#endif // _ANN_IO_READER_HPP
