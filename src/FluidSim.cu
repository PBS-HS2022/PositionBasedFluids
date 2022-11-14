#include "FluidSim.h"
#include <cuda_runtime.h>

#define HANDLE_CUDA_CALL(cuda_func, err) \
do {\
	cudaError_t err_code;\
	if((err_code = cuda_func) != cudaSuccess) { \
		cerr << "Cuda error in call: " << err  << " with error code: " << err_code << endl; \
		exit(1); \
	}\
} while(0)

// Computation depends on neighboring results, thus it can't be parallelized
void FluidSim::solvePoisson() {
	double dx2 = m_dx * m_dx;
	double residual = m_acc + 1; // initial residual
	double rho = 1;

	Array2d& p = p_pressure->x();

	for (int it = 0; residual > m_acc && it < m_iter; ++it) {
		// Note that the boundaries are handles by the framework, so you iterations should be similar to:
		for (int y = 1; y < m_res_y - 1; ++y) {
			for (int x = 1; x < m_res_x - 1; ++x) {
				double b = -p_divergence->x()(x, y) / m_dt * rho; // right-hand
				// TODO: update the pressure values
				p(x, y) = (dx2 * b +
					p(x - 1, y) + p(x + 1, y) +
					p(x, y - 1) + p(x, y + 1)) / 4.0;
				
			}
		}

		// Compute the new residual, i.e. the sum of the squares of the individual residuals (squared L2-norm)
		residual = 0;
		for (int y = 1; y < m_res_y - 1; ++y) {
			for (int x = 1; x < m_res_x - 1; ++x) {
				double b = -p_divergence->x()(x, y) / m_dt * rho; // right-hand
				// TODO: compute the cell residual
				double cellResidual = b - (4 * p(x, y) -
					p(x - 1, y) - p(x + 1, y) -
					p(x, y - 1) - p(x, y + 1)) / dx2;

				residual += cellResidual * cellResidual;

			}
		}

		// Get the L2-norm of the residual
		residual = sqrt(residual);

		// We assume the accuracy is meant for the average L2-norm per grid cell
		residual /= (m_res_x - 2) * (m_res_y - 2);

		//// For your debugging, and ours, please add these prints after every iteration
		//cout << "Pressure solver: iter=" << it << ", res=" << residual << endl;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////CUDA CORRECT_VELOCITY///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void correctVelocity_kernel_u(double* cu_p, double* cu_u, double m_dt_idx, int warpsPerBlock, int length) {
	// Compute current thread index
	int j = blockIdx.x + 1; // +1 accounts for the array indexing starting at 1
	int i = threadIdx.x * warpsPerBlock + 1; 

	int array_idx = i * length + j;

	// Loop for thread coalescing
	for (int l = 0; l < warpsPerBlock; ++l) {
		cu_u[array_idx] = cu_u[array_idx] - (m_dt_idx * (cu_p[array_idx]- cu_p[(i-1) * length + j]));
	}
}

__global__ void correctVelocity_kernel_v(double* cu_p, double* cu_v, double m_dt_idx, int warpsPerBlock, int length) {
	// Compute current thread index
	int j = blockIdx.x + 1; // +1 accounts for the array indexing starting at 1
	int i = threadIdx.x * warpsPerBlock + 1;

	int array_idx = i * length + j;

	// Loop for thread coalescing
	for (int l = 0; l < warpsPerBlock; ++l) {
		cu_v[array_idx] = cu_v[array_idx] - (m_dt_idx * (cu_p[array_idx] - cu_p[i * length + j-1]));
	}
}

// CUDA Correct Velocity
void FluidSim::correctVelocity() {
	Array2d& p = p_pressure->x();
	Array2d& u = p_velocity->x();
	Array2d& v = p_velocity->y();

	// Precompute for kernels
	double m_dt_idx = m_dt * m_idx;

	// Retrieve device property
	int max_thrds = prop.maxThreadsPerBlock;

	// Compute n_iterations for both u & v
	int length_u = (m_res_y - 1) * m_res_x;
	int length_v = m_res_y * (m_res_x - 1);

	// Compute CUDA-specific parallelization parameters
	int nBlks_u = length_u - 2;
	int nBlks_v = length_v - 2;

	// Clamp the number of threads to the maximum amount allowed by the device
	int threadsPerBlk_u = fmin(nBlks_u, max_thrds);
	int threadsPerBlk_v = fmin(nBlks_v, max_thrds);

	// Compute the number of threads per block
	int warpsPerBlock_u = ((nBlks_u * threadsPerBlk_u) + max_thrds - 1) / max_thrds;
	int warpsPerBlock_v = ((nBlks_v * threadsPerBlk_v) + max_thrds - 1) / max_thrds;

	// Prepare GPU variables
	double *cu_p, *cu_u, *cu_v;
	double* p_addr = &(p(0,0));
	double* u_addr = &(u(0,0));
	double* v_addr = &(v(0,0));

	int size_p = p.size(0) * p.size(1) * sizeof(double);
	int size_u = u.size(0) * u.size(1) * sizeof(double);
	int size_v = v.size(0) * v.size(1) * sizeof(double);

	// Try to allocate space for p, u, and v on GPU
	HANDLE_CUDA_CALL(cudaMalloc((void**)&cu_p, size_p), "Malloc p");
	HANDLE_CUDA_CALL(cudaMalloc((void**)&cu_u, size_u), "Malloc u");
	HANDLE_CUDA_CALL(cudaMalloc((void**)&cu_v, size_v), "Malloc v");

	// Copy current array data from host to device
	HANDLE_CUDA_CALL(cudaMemcpy(cu_p, p_addr, size_p, cudaMemcpyHostToDevice), "Cpy p to device");
	HANDLE_CUDA_CALL(cudaMemcpy(cu_u, u_addr, size_u, cudaMemcpyHostToDevice), "Cpy u to device");
	HANDLE_CUDA_CALL(cudaMemcpy(cu_v, v_addr, size_v, cudaMemcpyHostToDevice), "Cpy v to device");

	// Call kernels
	correctVelocity_kernel_u<<<nBlks_u, threadsPerBlk_u>>>(cu_p, cu_u, m_dt_idx, warpsPerBlock_u, length_u);
	correctVelocity_kernel_v<<<nBlks_v, threadsPerBlk_v>>>(cu_p, cu_v, m_dt_idx, warpsPerBlock_v, length_v);

	// Copy results back from the device to the host (p isn't modified so no need)
	HANDLE_CUDA_CALL(cudaMemcpy(u_addr, cu_u, size_u, cudaMemcpyDeviceToHost), "Cpy u to Host");
	HANDLE_CUDA_CALL(cudaMemcpy(v_addr, cu_v, size_v, cudaMemcpyDeviceToHost), "Cpy v to Host");

	// Free CUDA variables
	cudaFree(cu_p);
	cudaFree(cu_u);
	cudaFree(cu_v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////END: CUDA CORRECT_VELOCITY////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FluidSim::advectValues() {
	// store original values
	Array2d d(p_density->x());
	Array2d u(p_velocity->x());
	Array2d v(p_velocity->y());
	// move forward
	advectDensitySL(u, v);
	advectVelocitySL(u, v);

	if (m_macOn){
		// store forward advection
		Array2d d_forward(p_density->x());
		Array2d u_forward(p_velocity->x());
		Array2d v_forward(p_velocity->y());
		MacCormackUpdate(d, d_forward, u, u_forward, v, v_forward);
		MacCormackClamp(d, d_forward, u, u_forward, v, v_forward);
	}
}

void FluidSim::advectDensitySL(const Array2d& u, const Array2d& v) {
	Array2d& d = p_density->x();
	Array2d d_tmp(d.size(0), d.size(1));

	// Densities, grid centers
	for (int y = 1; y < m_res_y - 1; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = (u(x, y) + u(x + 1, y)) / 2;
			double last_y_velocity = (v(x, y) + v(x, y + 1)) / 2;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Densities are known between 1 and res-2
			if (last_x < 1) last_x = 1;
			if (last_y < 1) last_y = 1;
			if (last_x > m_res_x - 2) last_x = m_res_x - 2;
			if (last_y > m_res_y - 2) last_y = m_res_y - 2;

			// Determine corners for bilinear interpolation
			int x_low = (int)last_x;
			int y_low = (int)last_y;
			int x_high = x_low + 1;
			int y_high = y_low + 1;

			// Compute the interpolation weights
			double x_weight = last_x - x_low;
			double y_weight = last_y - y_low;

			// TODO: Bilinear interpolation
			d_tmp(x, y) = x_weight * y_weight * d(x_high, y_high) +
				(1 - x_weight) * y_weight * d(x_low, y_high) +
				x_weight * (1 - y_weight) * d(x_high, y_low) +
				(1 - x_weight) * (1 - y_weight) * d(x_low, y_low);

		}
	}

	// Copy the values in temp to the original buffers
	d = d_tmp;
}

void FluidSim::advectVelocitySL(const Array2d& u, const Array2d& v) {
	Array2d& u_in = p_velocity->x();
	Array2d& v_in = p_velocity->y();

	Array2d u_tmp(u_in.size(0), u_in.size(1));
	Array2d v_tmp(v_in.size(0), v_in.size(1));

	// Velocities (u), MAC grid
	for (int y = 1; y < m_res_y - 1; ++y) {
		for (int x = 1; x < m_res_x; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = u(x, y);
			double last_y_velocity = (v(x, y) + v(x - 1, y) + v(x - 1, y + 1) + v(x, y + 1)) / 4;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Being conservative, one can say that the velocities are known between 1.5 and res-2.5
			// (the MAC grid is inside the known densities, which are between 1 and res - 2)
			if (last_x < 1.5) last_x = 1.5;
			if (last_y < 1.5) last_y = 1.5;
			if (last_x > m_res_x - 1.5) last_x = m_res_x - 1.5;
			if (last_y > m_res_y - 2.5) last_y = m_res_y - 2.5;

			// Determine corners for bilinear interpolation
			int x_low = (int)last_x;
			int y_low = (int)last_y;
			int x_high = x_low + 1;
			int y_high = y_low + 1;

			// Compute the interpolation weights
			double x_weight = last_x - x_low;
			double y_weight = last_y - y_low;

			// TODO: Bilinear interpolation
			u_tmp(x, y) = x_weight * y_weight * u_in(x_high, y_high) +
				(1 - x_weight) * y_weight * u_in(x_low, y_high) +
				x_weight * (1 - y_weight) * u_in(x_high, y_low) +
				(1 - x_weight) * (1 - y_weight) * u_in(x_low, y_low);

		}
	}

	// Velocities (v), MAC grid
	for (int y = 1; y < m_res_y; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = (u(x, y) + u(x + 1, y) + u(x + 1, y - 1) + u(x, y - 1)) / 4;
			double last_y_velocity = v(x, y);

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Being conservative, one can say that the velocities are known between 1.5 and res-2.5
			// (the MAC grid is inside the known densities, which are between 1 and res - 2)
			if (last_x < 1.5) last_x = 1.5;
			if (last_y < 1.5) last_y = 1.5;
			if (last_x > m_res_x - 2.5) last_x = m_res_x - 2.5;
			if (last_y > m_res_y - 1.5) last_y = m_res_y - 1.5;

			// Determine corners for bilinear interpolation
			double x_low = (int)last_x;
			double y_low = (int)last_y;
			double x_high = x_low + 1;
			double y_high = y_low + 1;

			// Compute the interpolation weights
			double x_weight = last_x - x_low;
			double y_weight = last_y - y_low;

			// TODO: Bilinear interpolation
			v_tmp(x, y) = x_weight * y_weight * v_in(x_high, y_high) +
				(1 - x_weight) * y_weight * v_in(x_low, y_high) +
				x_weight * (1 - y_weight) * v_in(x_high, y_low) +
				(1 - x_weight) * (1 - y_weight) * v_in(x_low, y_low);
		}
	}

	// Copy the values in temp to the original buffers
	u_in = u_tmp;
	v_in = v_tmp;
}

void FluidSim::MacCormackUpdate(const Array2d& d, const Array2d& d_forward, const Array2d& u,  const Array2d& u_forward, const Array2d& v, const Array2d& v_forward) {
	// move backward
	m_dt *= -1;
	advectDensitySL(u, v);
	advectVelocitySL(u, v);

	// store backward advection
	Array2d d_backward(p_density->x());
	Array2d u_backward(p_velocity->x());
	Array2d v_backward(p_velocity->y());
	m_dt *= -1;

	Array2d d_tmp(d_forward);
	Array2d u_tmp(u_forward);
	Array2d v_tmp(v_forward);
	// MacCormack Update
	for (int y = 1; y < m_res_y - 1; ++y)
		for (int x = 1; x < m_res_x - 1; ++x)
			// TODO: update d
			d_tmp(x, y) += 0.5 * (d(x, y) - d_backward(x, y));

	for (int y = 1; y < m_res_y - 1; ++y)
		for (int x = 1; x < m_res_x; ++x)
			// TODO: update u
			u_tmp(x, y) += 0.5 * (u(x, y) - u_backward(x, y));
		
	for (int y = 1; y < m_res_y; ++y)
		for (int x = 1; x < m_res_x - 1; ++x)
			// TODO: update v
			v_tmp(x, y) += 0.5 * (v(x, y) - v_backward(x, y));
	
	p_density->x() = d_tmp;
	p_velocity->x() = u_tmp;
	p_velocity->y() = v_tmp;
}

void FluidSim::MacCormackClamp(const Array2d& d, const Array2d& d_forward, const Array2d& u,  const Array2d& u_forward, const Array2d& v, const Array2d& v_forward){
	Array2d d_tmp(p_density->x());
	Array2d u_tmp(p_velocity->x());
	Array2d v_tmp(p_velocity->y());
	// Clamp density
	for (int y = 1; y < m_res_y - 1; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			double last_x_velocity = (u(x, y) + u(x + 1, y)) / 2;
			double last_y_velocity = (v(x, y) + v(x, y + 1)) / 2;

			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Densities are known between 1 and res-2
			if (last_x < 1) last_x = 1;
			if (last_y < 1) last_y = 1;
			if (last_x > m_res_x - 2) last_x = m_res_x - 2;
			if (last_y > m_res_y - 2) last_y = m_res_y - 2;

			// Determine corners for bilinear interpolation
			int x_low = (int)last_x;
			int y_low = (int)last_y;
			int x_high = x_low + 1;
			int y_high = y_low + 1;

			double d_min = 1e10;
			double d_max = -1e10;

			d_min = min(d(x_low, y_low), d_min);
			d_min = min(d(x_low, y_high), d_min);
			d_min = min(d(x_high, y_low), d_min);
			d_min = min(d(x_high, y_high), d_min);

			d_max = max(d(x_low, y_low), d_max);
			d_max = max(d(x_low, y_high), d_max);
			d_max = max(d(x_high, y_low), d_max);
			d_max = max(d(x_high, y_high), d_max);

			// TODO: clamp d
			if (d_tmp(x, y) < d_min || d_tmp(x, y) > d_max)
				d_tmp(x, y) = d_forward(x, y);
				
		}
	}

	// Clamp velocities (u), MAC grid
	for (int y = 1; y < m_res_y - 1; ++y) {
		for (int x = 1; x < m_res_x; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = u(x, y);
			double last_y_velocity = (v(x, y) + v(x - 1, y) + v(x - 1, y + 1) + v(x, y + 1)) / 4;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Being conservative, one can say that the velocities are known between 1.5 and res-2.5
			// (the MAC grid is inside the known densities, which are between 1 and res - 2)
			if (last_x < 1.5) last_x = 1.5;
			if (last_y < 1.5) last_y = 1.5;
			if (last_x > m_res_x - 1.5) last_x = m_res_x - 1.5;
			if (last_y > m_res_y - 2.5) last_y = m_res_y - 2.5;

			// Determine corners for bilinear interpolation
			int x_low = (int)last_x;
			int y_low = (int)last_y;
			int x_high = x_low + 1;
			int y_high = y_low + 1;

			double u_min = 1e10;
			double u_max = -1e10;

			u_min = min(u(x_low, y_low), u_min);
			u_min = min(u(x_low, y_high), u_min);
			u_min = min(u(x_high, y_low), u_min);
			u_min = min(u(x_high, y_high), u_min);

			u_max = max(u(x_low, y_low), u_max);
			u_max = max(u(x_low, y_high), u_max);
			u_max = max(u(x_high, y_low), u_max);
			u_max = max(u(x_high, y_high), u_max);

			// TODO: clamp u
			if (u_tmp(x, y) < u_min || u_tmp(x, y) > u_max)
				u_tmp(x, y) = u_forward(x, y);

		}
	}

	// Clamp velocities (v), MAC grid
	for (int y = 1; y < m_res_y; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = (u(x, y) + u(x + 1, y) + u(x + 1, y - 1) + u(x, y - 1)) / 4;
			double last_y_velocity = v(x, y);

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = x - m_dt * m_idx * last_x_velocity;
			double last_y = y - m_dt * m_idx * last_y_velocity;

			// Make sure the coordinates are inside the boundaries
			// Being conservative, one can say that the velocities are known between 1.5 and res-2.5
			// (the MAC grid is inside the known densities, which are between 1 and res - 2)
			if (last_x < 1.5) last_x = 1.5;
			if (last_y < 1.5) last_y = 1.5;
			if (last_x > m_res_x - 2.5) last_x = m_res_x - 2.5;
			if (last_y > m_res_y - 1.5) last_y = m_res_y - 1.5;

			// Determine corners for bilinear interpolation
			double x_low = (int)last_x;
			double y_low = (int)last_y;
			double x_high = x_low + 1;
			double y_high = y_low + 1;
			
			double v_min = 1e10;
			double v_max = -1e10;

			v_min = min(v(x_low, y_low), v_min);
			v_min = min(v(x_low, y_high), v_min);
			v_min = min(v(x_high, y_low), v_min);
			v_min = min(v(x_high, y_high), v_min);

			v_max = max(v(x_low, y_low), v_max);
			v_max = max(v(x_low, y_high), v_max);
			v_max = max(v(x_high, y_low), v_max);
			v_max = max(v(x_high, y_high), v_max);

			// TODO: clamp v
			if (v_tmp(x, y) < v_min || v_tmp(x, y) > v_max)
				v_tmp(x, y) = v_forward(x, y);

		}
	}
	
	p_density->x() = d_tmp;
	p_velocity->x() = u_tmp;
	p_velocity->y() = v_tmp;
}