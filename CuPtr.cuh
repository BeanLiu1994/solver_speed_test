#pragma once
#include <cuda_runtime.h>
#include "Culocator.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <typeinfo>

template<typename _tyIn>
class CuPtr
{
	typedef typename std::decay<_tyIn>::type _ty;
	typedef typename std::remove_reference<_tyIn>::type _ty_noref;
public:
	CuPtr(size_t _size, _ty initVal = _ty()) :
		type_size(sizeof(_ty)),
		elem_size(_size),
		mem_size(_size * sizeof(_ty)),
		device_ptr(nullptr),
		ptr(nullptr)
	{
		if (mem_size == 0)
		{
			std::cout <<
				std::string("You're setting a  0 size of CuPtr<") + typeid(_ty).name() + ">, Please check your parameter if it's wrong.  CuPtr<...>(size, value)" 
					<< std::endl;
			return;
		}
		_cu_malloc(&device_ptr, mem_size);
		_cu_memset(device_ptr, mem_size, initVal);
	}

	CuPtr(size_t _size, _ty_noref* _ptr) :
		type_size(sizeof(_ty)),
		elem_size(_size),
		mem_size(_size * sizeof(_ty)),
		device_ptr(nullptr),
		ptr(_ptr)
	{
		_cu_malloc(&device_ptr, mem_size);
		if (ptr != nullptr)
			_cu_copyToDevice(device_ptr, mem_size, ptr);
		else
			_cu_memset(device_ptr, mem_size, 0);
	}
	
	_ty* operator()() 
	{
		return static_cast<_ty*>(device_ptr);
	}
	const _ty* operator()() const
	{
		return static_cast<_ty*>(device_ptr);
	}

	void GetResult(_ty* OutPtr = nullptr)
	{
		if (std::is_same<_ty_noref, _ty>::value)
		{
			if (OutPtr != nullptr)
				ptr = OutPtr;
			if (ptr == nullptr)
				throw std::runtime_error("null pointer");
			_cu_getResult(device_ptr, mem_size, ptr);
		}
		else
		{
			std::cerr << "can't run GetResult for const type." << std::endl;
		}
	}

	~CuPtr()
	{
		_cu_free(device_ptr);
	}

	static void SyncDevice()
	{
		_cu_syncDevice();
	}
	
	size_t Get_type_size(){return type_size;}
	size_t Get_elem_size(){return elem_size;}
	size_t Get_mem_size(){return mem_size;}

protected:
	_ty_noref * ptr;
	void* device_ptr;
	const size_t type_size;
	size_t elem_size, mem_size;

	CuPtr(const CuPtr&) = delete;
	CuPtr operator=(const CuPtr&) = delete;
};


