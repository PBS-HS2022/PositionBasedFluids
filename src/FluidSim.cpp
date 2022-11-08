#include "FluidSim.h"

/////////////////////////////////////
/////////////// EX 4 ////////////////
/////////////////////////////////////
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
				
			}
		}

		// Compute the new residual, i.e. the sum of the squares of the individual residuals (squared L2-norm)
		residual = 0;
		for (int y = 1; y < m_res_y - 1; ++y) {
			for (int x = 1; x < m_res_x - 1; ++x) {
				double b = -p_divergence->x()(x, y) / m_dt * rho; // right-hand
				// TODO: compute the cell residual
				double cellResidual = 0.0;

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

void FluidSim::correctVelocity() {
	Array2d& p = p_pressure->x();
	Array2d& u = p_velocity->x();
	Array2d& v = p_velocity->y();

	// Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
	for (int y = 1; y < m_res_y - 1; ++y)
		for (int x = 1; x < m_res_x; ++x)
			// TODO: update u
			u(x, y) = u(x, y);

	// Same for velocity v_{i+1/2}.
	for (int y = 1; y < m_res_y; ++y)
		for (int x = 1; x < m_res_x - 1; ++x)
			// TODO: update v
			v(x, y) = v(x, y);
}

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
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = 0.;
			double last_y = 0.;

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
			d_tmp(x, y) = d(x, y);
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
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = 0.;
			double last_y = 0.;

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
			u_tmp(x, y) = u(x, y);
		}
	}

	// Velocities (v), MAC grid
	for (int y = 1; y < m_res_y; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = 0.;
			double last_y = 0.;

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
			v_tmp(x, y) = v(x, y);
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
			d_tmp(x, y) = d_tmp(x, y);

	for (int y = 1; y < m_res_y - 1; ++y)
		for (int x = 1; x < m_res_x; ++x)
			// TODO: update u
			u_tmp(x, y) = u_tmp(x, y);
		
	for (int y = 1; y < m_res_y; ++y)
		for (int x = 1; x < m_res_x - 1; ++x)
			// TODO: update v
			v_tmp(x, y) = v_tmp(x, y);
	
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
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

			double last_x = 0.;
			double last_y = 0.;

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
			d_tmp(x, y) = d_tmp(x, y);
				
		}
	}

	// Clamp velocities (u), MAC grid
	for (int y = 1; y < m_res_y - 1; ++y) {
		for (int x = 1; x < m_res_x; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

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
			u_tmp(x, y) = u_tmp(x, y);
		}
	}

	// Clamp velocities (v), MAC grid
	for (int y = 1; y < m_res_y; ++y) {
		for (int x = 1; x < m_res_x - 1; ++x) {
			// TODO: Compute the velocity
			double last_x_velocity = 0.;
			double last_y_velocity = 0.;

			// TODO: Find the last position of the particle (in grid coordinates)
			double last_x = 0.;
			double last_y = 0.;

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
			v_tmp(x, y) = v_tmp(x, y);
		}
	}
	
	p_density->x() = d_tmp;
	p_velocity->x() = u_tmp;
	p_velocity->y() = v_tmp;
}