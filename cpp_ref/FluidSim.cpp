#include "FluidSim.h"
#include <algorithm>

// +++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++ SPH FUNCTIONS ++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++

// =================================================
// =========== Initialize SPH Particles ============
// =================================================
void FluidSim::initSPH(double xmin, double xmax, double ymin, double ymax) {
	m_xmin = xmin;
	m_xmax = xmax;
	m_ymin = ymin;
	m_ymax = ymax;
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
// =============== Find Neighbors ==================
// =================================================
// Finds neighbors for all particles at the current time.
std::vector<std::vector<int>> FluidSim::findNeighbors() {
	// Implementation of Blazing-Fast Neighbor Searching with Spatial Hashing
	// https://matthias-research.github.io/pages/tenMinutePhysics/11-hashing.pdf
	auto get_grid_hash = [this](Eigen::Vector2i grid_pos) {
		int h = (grid_pos.x() * 92837111) ^ (grid_pos.y() * 689287499);
		// Add the following term for 3D:
		// ^ (grid_pos.z() * 283923481);
		return abs(h) % particles.size();
	};

	auto get_nearest_grid = [this](Eigen::Vector2d particle_pos) {
		// Use Vector3i for 3D:
		// return Eigen::Vector3i((int)floor(particle_pos.x() / m_dx), (int)floor(particle_pos.y() / m_dx), (int)floor(particle_pos.z() / m_dx));
		return Eigen::Vector2i((int)floor(particle_pos.x() /(m_h * 1.5)), (int)floor(particle_pos.y() / (m_h * 1.5)));
	};

	// 2D Hashtable representing spatial coordinates. The value in each cell
	// is a vector of all particle (global indices) that exist in that grid
	// location. If there are no particles at that grid cell, the vector is of
	// size zero.
	std::vector<std::vector<int>> splatted_grid(particles.size());

	for (int i = 0; i < particles.size(); i++) {
		Particle p_i = particles[i];
		// Find the spatial hash and append the particle global index to the
		// vector at that collection grid.
		splatted_grid[get_grid_hash(get_nearest_grid(p_i.x))].push_back(i);
	}

	// Now go through each particle, find the spatial hash, and get the other
	// particles at the same grid location, collected above. This will include
	// the particle itself as the first entry.
	std::vector<std::vector<int>> neighbors(particles.size());
	for (int i = 0; i < particles.size(); i++) {
		Particle p_i = particles[i];

		// Find nearest grid positions
		Eigen::Vector2i grid_pos = get_nearest_grid(p_i.x);
		for (int x : {grid_pos.x() - 1, grid_pos.x(), grid_pos.x() + 1}) {
			for (int y : {grid_pos.y() - 1, grid_pos.y(), grid_pos.y() + 1}) {
				// For all particles that are in this cell, collect those that
				// are within the defined proximity radius, which is the cell
				// size.
				// Things that may fail this test include points that are at
				// opposite diagonal ends in a cell.
				auto this_grid_hash = get_grid_hash(Eigen::Vector2i(x, y));
				for (int j : splatted_grid[this_grid_hash]) {
					if ((particles[j].x - p_i.x).squaredNorm() < m_dx) {
						neighbors[i].push_back(j);
					}
				}
			}
		}
	}

	return neighbors;
}


// =================================================
// ============== Solve Fluids =====================
// =================================================
void FluidSim::solveFluids() {
	std::vector<std::vector<int>> neighbors = findNeighbors();
	for (int i=0; i < particles.size(); i++) {
		Particle p_i = particles[i];

		// Initial values
		float rho = 0.0f;
		float sum_grad2 = 0.0f;
		Eigen::Vector2d grad_i(0.0f, 0.0f);
		
		// Loop through neighbors
		for (int neighbor_ix=0; neighbor_ix < neighbors[i].size(); neighbor_ix++) {
			int id = neighbors[i][neighbor_ix];
			Particle p_j = particles[id];
			//Calculate distance between particles
			Eigen::Vector2d n = p_j.x - p_i.x; 
			float r = n.norm();

			// normalize
			if (r > 0.0f) {
				n /= r;
			}
			// If distance is greater than kernel radius (h)
			if (r > m_h) {
				m_grads[neighbor_ix] = Eigen::Vector2d(0.0f);
			}
			else {
				float r2 = r * r;
				float w = (m_h * m_h) - r2;
				
				rho += m_POLY6 * w * w * w;
				float grad = (m_POLY6 * 3.0f * w * w * (-2.0f * r)) / m_rho0;

				m_grads[neighbor_ix] = n * grad;

				grad_i -= n * grad;
				sum_grad2 += grad * grad;
			}
		}

		sum_grad2 += grad_i.squaredNorm();

		float c = rho / m_rho0 - 1.0f;
		if (c < 0.0f) {
			continue;
		}
		
		float lambda = -c / (sum_grad2 + 0.0001);

		for (int neighbor_ix=0; neighbor_ix < neighbors[i].size(); neighbor_ix++) {
			int id = neighbors[i][neighbor_ix];
			Particle p_j = particles[id];

			if (id == i) {
				p_j.x += lambda * grad_i;
			}
			else {
				p_j.x += lambda * m_grads[neighbor_ix];
			}
		}

	}
}

// =================================================
// ============= Boundary Checks ===================
// =================================================
void FluidSim::solveBoundaries() {
	for (auto &p_i : particles) {
		// Clamp positions to edges if it goes beyond
		if (p_i.x(1) < 0.0f) {
			p_i.x(1) = 0.0f;
		}
		p_i.x(0) = std::clamp(p_i.x(0), 0.0, (double)m_res_x);

		// More boundary checks?
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
	// cout << "FIRST LOOP" << endl;
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

	// TODO : BOUNDARY CHECKING HERE

	// cout << "CONSTRAINT LOOP" << endl;
	// Iterate to solve the constraints
	for (int iteration = 0; iteration < getIteration(); iteration++) {
		//Euler step? Maybe...

		// Solve fluids here
		solveBoundaries();
		solveFluids();

		// For each particle, calculate lambda
		// cout << "ITeration: " << iteration << endl;
		// for (int i = 0; i < particles.size(); i++) {
		// 	Particle p_i = particles[i];

		// 	// Calculate the gradient of the constraint function with respect
		// 	// to all (k) particles, and use its squared sum. Equation 9 of the
		// 	// PBF paper. In the PBF paper, k refers to all particles, but some
		// 	// publicly available implementations use just neighboring particles.
		// 	float sqrd_grad_constraints = 0;
		// 	// Note: for now we naively use all particles as its neighbours
		// 	// cout << "PARTICLE I : " << i << endl;
		// 	for (int l = 0; l < neighbour_indices[i].size(); l++) {
		// 		int k = neighbour_indices[i][l];
		// 		Particle p_k = particles[k];
		// 		// cout << "PARTICLE K" << k << endl;
		// 		// Equation 7 & 8 in the PBF paper, finding the gradient of the
		// 		// constraint function (for particle i) with respect with
		// 		// particle k.
		// 		if (k == i) {
		// 			// k == i case
		// 			// Loop over neighbours of i, call their indices j.
		// 			float sum = 0;
		// 			for (int j = 0; j < neighbour_indices[i].size(); j++) {
		// 				// cout << "Particle J : " << j << endl;
		// 				// cout << neighbour_indices[i][j] << " -> " << particles.size() << endl;
		// 				Particle p_j = particles[neighbour_indices[i][j]];
		// 				// cout << "After particle access " << endl;
		// 				// TODO: is .norm() the right thing to do?
		// 				sum += (p_i.x - p_j.x).norm() * m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f);
		// 			}
		// 			sqrd_grad_constraints += pow(1 / (m_rho0) * sum, 2);
		// 		} else {
		// 			// k == j case
		// 			Particle p_j = particles[k];
		// 			// TODO: is .norm() the right thing to do?
		// 			sqrd_grad_constraints += pow((p_i.x - p_j.x).norm() * 1 / (m_rho0) * -(m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f)), 2);
		// 		}
		// 	}

		// 	lambda[i] = -density_constraints[i] / sqrd_grad_constraints;
		// }

		// // cout << "DELTA P LOOP" << endl;
		// for (int i = 0; i < particles.size(); i++) {
		// 	Particle p_i = particles[i];

		// 	// calculate delta_p eq12

		// 	delta_x[i] = Eigen::Vector2d();
		// 	// Loop over the neighbouring particles as per Equation 12 in PBF.
		// 	for (int j = 0; j < neighbour_indices[i].size(); j++) {
		// 		if (neighbour_indices[i][j] == i) continue;
		// 		Particle p_j = particles[neighbour_indices[i][j]];

		// 		// Any correction term (defined in Equation 13, unimplemented)
		// 		float corr = 0;

		// 		// cout << (p_i.x - p_j.x).normalized() << " ==> " << p_i.x << " --> " << p_j.x << endl;

		// 		delta_x[i] += (p_i.x - p_j.x).normalized() * ((1 / m_rho0) * (lambda[i] + lambda[j] + corr) * m_SPIKY_GRAD * pow(p_i.x.norm() - p_j.x.norm(), 3.0f));
		// 	}
		// 	// TODO: any collision detections go here
		// }

		// for (int i = 0; i < particles.size(); i++) {
		// 	Particle p_i = particles[i];

		// 	// calculate new_x based on delta_x
		// 	new_x[i] = p_i.x + delta_x[i];
		// }
	}

	// Correct the velocities based on the new constraint-solved positions
	// And confirm the positions
	int ctr = 0;
	// cout << "FINAL LOOP" << endl;
	for (auto &p : particles) {
		// cout << "START OF LOOP: " << p_i.x << endl;
		p.v = 1 / m_dt * (new_x[ctr] - p.x);
		p.x = new_x[ctr];
		cout << "New position of particle " << ctr << "/" << particles.size() << ": " << p.x.transpose().format(Eigen::IOFormat(2, 0, ", ")) << " and its velocity: " << p.v.transpose().format(Eigen::IOFormat(2, 0, ", ")) << endl;
		ctr++;

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
		// std::cout << p.x << std::endl;
		std::cout << "%: " << abs((int)p.x.x()) % m_res_x << " " << abs((int)p.x.y()) % m_res_y << std::endl;
		std::cout << "From .cpp: " << m_res_x << " " << m_res_y << std::endl;
		p_density->set_m_x(abs((int)p.x.x()) % m_res_x, abs((int)p.x.y()) % m_res_y);
	}
	// yuto: removed stuff down here for debugging, check git log
	std::cout << "here" << std::endl;

	// p_density->x() = d_tmp;
}
