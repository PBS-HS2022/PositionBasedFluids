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

void FluidSim::correctVelocity() {
	Array2d& p = p_pressure->x();
	Array2d& u = p_velocity->x();
	Array2d& v = p_velocity->y();

	// Note: velocity u_{i+1/2} is practically stored at i+1, hence xV_{i}  -= dt * (p_{i} - p_{i-1}) / dx
	for (int y = 1; y < m_res_y - 1; ++y)
		for (int x = 1; x < m_res_x; ++x)
			// TODO: update u
			u(x, y) = u(x, y) - (m_dt * (p(x, y) - p(x-1, y)) * m_idx);

	// Same for velocity v_{i+1/2}.
	for (int y = 1; y < m_res_y; ++y)
		for (int x = 1; x < m_res_x - 1; ++x)
			// TODO: update v
			v(x, y) = v(x, y) - (m_dt * (p(x, y) - p(x, y-1)) * m_idx);
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

// +++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++ SPH FUNCTIONS ++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++

// =================================================
// =========== Initialize SPH Particles ============
// =================================================
void FluidSim::initSPH(double xmin, double xmax, double ymin, double ymax) {
	// for (float y = 16.0f; y < m_res_y - 16.0f * 2.0f; y += 16.0f) {
	// 	for (float x = m_res_x / 4; x <= m_res_x / 2; x += 16.0f) {
	for (int y = (int)(ymin * m_res_y); y < (int)(ymax * m_res_y); y++) {
		for (int x = (int)(xmin * m_res_x); x < (int)(xmax * m_res_x); x++) {
			if (particles.size() < m_NUM_PARTICLES) {
				float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
				Particle newParticle(x + jitter, y);
				particles.push_back(newParticle);
				p_density->set_m_x(newParticle.x.x(), newParticle.x.y());
			}
			else {
				return;
			}
		}
	}
}


// =================================================
// ==== Compute Densities and Pressures for SPH ====
// =================================================
void FluidSim::computePressureSPH() {
	float h2 = m_h * m_h;
	for (auto &pi : particles) {
		pi.rho = 0.0f;

		for (auto &pj : particles) {
			Eigen::Vector2d r_ij = pj.x - pi.x;
			float r2 = r_ij.squaredNorm();

			if (r2 < h2) {
				pi.rho += m_mass * m_POLY6 * pow(h2 - r2, 3.0f);
			}
		}

		// pi.rho *= 1000.0f;

		//pi.p = std::max(m_k * (((float)pow(pi.rho, 7) / (float)pow(m_rho0, 7)) - 1), 0.0f);
		pi.p = std::max(m_k * (pi.rho - m_rho0), 0.0f);
		// pi.p = m_k * (pi.rho - m_rho0);
	}
}

// =================================================
// ============ Compute forces for SPH =============
// =================================================
void FluidSim::computeForcesSPH() {
	for (auto &pi : particles) {
		Eigen::Vector2d f_p(0.0f, 0.0f);
		Eigen::Vector2d f_v(0.0f, 0.0f);

		for (auto &pj : particles) {
			if (&pi == &pj) {
				continue;
			}

			Eigen::Vector2d r_ij = pj.x - pi.x;
			float r = r_ij.norm();

			if (r < m_h) {
				// Pressure forces
				f_p += -r_ij.normalized() * m_mass * (pi.p + pj.p) / (2.0f * pj.rho) * m_SPIKY_GRAD * pow(m_h - r, 3.0f);
				// f_p += -r_ij.normalized() * m_mass * ((pi.p / pow(pi.rho,2)) + (pj.p / pow(pj.rho,2))) * m_SPIKY_GRAD * pow(m_h - r, 3.0f);

				// Viscosity forces
				f_v += m_visc_cons * m_mass * (pj.v - pi.v) / pj.rho * m_VISC_LAP * (m_h - r);
			}			
		}

		// Gravitational force
		Eigen::Vector2d f_g = pi.rho * m_G;

		// cout << "Time: " << m_time << "density: " << pi.rho << " pressure: " << f_p << " viscosity:" << f_v << " gravity: " << f_g << endl;
		
		// Add up all the forces
		pi.f = f_p + f_v + f_g;
	}
}

// =================================================
// ===== SPH Integration and boundary checks =======
// =================================================
void FluidSim::integrateSPH() {
	// Array2d d_tmp(p_density->x());
	p_density->reset();

	// Values of C for each particle, calculated with equation 1 in PBF
	std::vector<float> density_constraints(particles.size());

	// Values of lambda for each particle, calculated with equation 8 and 9 in PBF
	std::vector<float> lambda(particles.size());

	// The change in position, for each particle
	std::vector<Eigen::Vector2d> delta_x(particles.size());

	// Values of the new positions for each particle, to be compared with the current one
	std::vector<Eigen::Vector2d> new_x(particles.size());

	// Wondering if for_each is better than an explicit for loop
	for (int i = 0; i < particles.size(); i++) {
		Particle p_i = particles[i];

		// Symplectic euler step with damping
		p_i.v += m_dt * p_i.f / p_i.rho;
		p_i.x += m_dt * p_i.v;

		// cout << "Time: " << m_time << " p.f: " << p.f << " p.rho:" << p.rho << endl;
		// cout << "Time: " << m_time << " p.v: " << p.v << " p.x: " << p.x << endl;

		// Create the density constraint for each particle
		density_constraints[i] = (p_i.rho - m_rho0) - 1;
	}

	// Iterate to solve the constraints
	for (int iteration = 0; iteration < 10; iteration++) {

		// For each particle, calculate lambda
		for (int i = 0; i < particles.size(); i++) {
			Particle p_i = particles[i];

			// Calculate the gradient of the constraint function with respect
			// to all (k) particles, and use its squared sum. Equation 9 of the
			// PBF paper. In the PBF paper, k refers to all particles, but some
			// publicly available implementations use just neighboring particles.
			float sqrd_grad_constraints = 0;
			// Note: for now we naively use all particles as its neighbours
			for (int k = 0; k < particles.size(); k++) {
				Particle p_k = particles[k];
				// Equation 7 & 8 in the PBF paper, finding the gradient of the
				// constraint function (for particle i) with respect with
				// particle k.
				if (k == i) {
					// k == i case
					// Loop over neighbours j. For now we naively use all particles.
					float sum = 0;
					for (int j = 0; j < particles.size(); j++) {
						Particle p_j = particles[j];
						// TODO: is .norm() the right thing to do?
						sum += m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f);
					}
					sqrd_grad_constraints += pow(1 / (m_rho0) * sum, 2);
				} else {
					// k == j case
					Particle p_j = particles[k];
					// TODO: is .norm() the right thing to do?
					sqrd_grad_constraints += pow(1 / (m_rho0) * -(m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f)), 2);
				}
			}

			lambda[i] = -density_constraints[i] / sqrd_grad_constraints;
		}

		for (int i = 0; i < particles.size(); i++) {
			Particle p_i = particles[i];

			// calculate delta_p eq12
			
			// Loop over the neighbouring particles as per Equation 12 in PBF.
			// TODO: change this to neighbouring particles instead of all
			for (int j = 0; j < particles.size(); j++) {
				Particle p_j = particles[j];

				// Any correction term (defined in Equation 13, unimplemented)
				float corr = 0;

				// TODO: this is wrong, because we want the gradient of the SPIKY
				// kernel and that should be returning a vector
				delta_x[i] += (p_i.x - p_j.x).normalized() * ((1 / m_rho0) * (lambda[i] + lambda[j] + corr) * m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f));
			}

			// TODO: any collision detections go here
		}

		for (int i = 0; i < particles.size(); i++) {
			Particle p_i = particles[i];

			// calculate new_x based on delta_x
			new_x[i] = p_i.x + delta_x[i];
		}
	}

	// Fix the velocities based on the new constraint-solved positions
	// And confirm the positions
	int ctr = 0;
	for (auto &p_i : particles) {
		p_i.v = 1 / m_dt * (new_x[ctr] - p_i.x);
		p_i.x = new_x[ctr];
		ctr++;
	}


	for (auto &p : particles) {
		// =================================================
		// ================ boundary checks ================
		// =================================================
		const float DAMP = 0.75;

		int x_coord = (int)p.x.x();
		int y_coord = (int)p.x.y();

		// Left & Right Walls
		if (p.v(0) != 0.0f) {
			// Left 
			if (x_coord < m_h) {
				float tbounce = (p.x(0) - m_h) / p.v(0);

				p.x(0) -= p.v(0) * (1 - DAMP) * tbounce;
				p.x(1) -= p.v(1) * (1 - DAMP) * tbounce;

				p.x(0) = 2 * m_h - p.x(0);
				p.v(0) = -p.v(0) * DAMP;
				p.v(1) *= DAMP;

				// p.v(0) *= DAMP;
				// p.x(0) = m_h;
			}
		}

		if (p.v(0) != 0.0f) {
			// Right
			if (x_coord > m_res_x - m_h) {
				float tbounce = (p.x(0) - m_res_x + m_h) / p.v(0);

				p.x(0) -= p.v(0) * (1 - DAMP) * tbounce;
				p.x(1) -= p.v(1) * (1 - DAMP) * tbounce;

				p.x(0) = 2*(m_res_x - m_h) - p.x(0);
				p.v(0) = -p.v(0) * DAMP;
				p.v(1) *= DAMP;

				// p.v(0) *= DAMP;
				// p.x(0) = m_res_x - m_h;
			}
		}	

		// Bottom and Top Boundaries
		if (p.v(1) != 0.0f) {
			// Bottom
			if (y_coord < m_h) {
				float tbounce = (p.x(1) - m_h) / p.v(1);

				p.x(0) -= p.v(0) * (1 - DAMP) * tbounce;
				p.x(1) -= p.v(1) * (1 - DAMP) * tbounce;

				p.x(1) = 2 * m_h - p.x(1);
				p.v(1) = -p.v(1) * DAMP;
				p.v(0) *= DAMP;

				// p.v(1) *= DAMP;
				// p.x(1) = m_h;
			}
		}

		if (p.v(1) != 0.0f) {
			// Top
			if (y_coord > m_res_y - m_h) {
				float tbounce = (p.x(1) - m_res_x + m_h) / p.v(1);

				p.x(0) -= p.v(0) * (1 - DAMP) * tbounce;
				p.x(1) -= p.v(1) * (1 - DAMP) * tbounce;

				p.x(1) = 2*(m_res_y - m_h) - p.x(1);
				p.v(1) = -p.v(1) * DAMP;
				p.v(0) *= DAMP;

				// p.v(1) *= DAMP;
				// p.x(1) = m_res_y - m_h;
			}
		}

		// Update grid positions
		p_density->set_m_x((int)p.x.x(), (int)p.x.y());
	}
	// p_density->x() = d_tmp;
}
