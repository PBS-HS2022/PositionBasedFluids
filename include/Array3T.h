#pragma once

#include <vector>
#include <assert.h>

// Simple 3D array - copied from Array2T. Note to Yuto: how does this differ from Eigen::Vector3D?
template <typename SCALAR>
class Array3T
{
public:
	// Default constructor
	Array3T()
	{
		m_size[0] = 0;
		m_size[1] = 0;
		m_size[2] = 0;
	}

	// Constructor with given size
	Array3T(int size0, int size1, int size2, SCALAR value = (SCALAR)0)
	{
		resize(size0, size1, size2, value);
	}

	// Copy constructor
	Array3T(const Array3T<SCALAR> &m)
	{
		*this = m;
	}

	// Resize array
	void resize(int size0, int size1, int size2, SCALAR value = (SCALAR)0)
	{
		m_size[0] = size0;
		m_size[1] = size1;
		m_size[2] = size2;

		m_data.resize(size0 * size1 * size2, value);
	}

	// Fill array with scalar s
	void fill(SCALAR s)
	{
		std::fill(m_data.begin(), m_data.end(), s);
	}

	// Fill array with 0
	void zero()
	{
		fill(0);
	}

	// Read & write element access
	SCALAR& operator()(unsigned int i, unsigned int j, unsigned int k)
	{
		assert(i >= 0 && i < m_size[0] && j >= 0 && j < m_size[1] && k >= 0 && k < m_size[2]);
		return m_data[i * m_size[1] * m_size[2] + j * m_size[2] + k];
	}

	// Read only element access
	const SCALAR& operator()(unsigned int i, unsigned int j, unsigned int k) const
	{
		assert(i >= 0 && i < m_size[0] && j >= 0 && j < m_size[1] && k >= 0 && k < m_size[2]);
		return m_data[i * m_size[1] * m_size[2] + j * m_size[2] + k];
	}

	// Dimension
	int size(int dimension) const
	{
		assert(dimension >= 0 && dimension < 3);
		return (int)m_size[dimension];
	}

	// Assignment
	Array3T<SCALAR> &operator=(const Array3T<SCALAR> &m3)
	{
		if (&m3 != this)
		{
			resize(m3.size(0), m3.size(1), m3.size(2));

			int n = (int)m_data.size();
			for (int i = 0; i < n; i++)
				m_data[i] = m3.m_data[i];
		}

		return *this;
	}

protected:
	unsigned int		m_size[3];
	std::vector<SCALAR>	m_data;
};

typedef Array3T<double> Array3d;
typedef Array3T<std::vector<int>> NeighborArray3d;
