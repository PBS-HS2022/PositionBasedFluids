#include <igl/edges.h>
#include "Simulation.h"
#include "Grid2.h"
#include "MACGrid2.h"

using namespace std;

// Struct for water particles
// Set initial position, velocity, force, density, pressure
struct Particle {
	Particle(float _x, float _y) 
		: x(_x, _y), v(0.0f,0.0f), f(0.0f, 0.0f), rho(0), p(0.0f) {}
	
	Eigen::Vector2d x, v, f;
	float rho, p;
};

/*
 * Simulation of a simple smoke plume rising.
 */
class FluidSim : public Simulation {
public:
	FluidSim() : Simulation() { init(); }

	virtual void init() override {
		m_res_x = 128;
		m_res_y = int(m_res_x*1.5); // 3:2 ratio
		m_size_x = m_res_x; // or just 1.0
		m_dx = m_size_x / m_res_x; // ! dx == dy
		m_idx = m_res_x / m_size_x;
		m_size_y = m_dx * m_res_y;
		m_dt = 0.005 * sqrt((m_res_x + m_res_y) * 0.5);
		// m_dt = 0.01;
		m_acc = 1e-5;
		m_iter = 1000;
		m_field = 0;
		m_velocityOn = false;
		m_vScale = 20;
		m_windOn = false;
		m_macOn = true;

		// ++++++++++ SPH variables +++++++++++++++++++++++
		m_NUM_PARTICLES = 2500;

		m_mass = 2.5f;
		m_k = 100.0f;
		m_h = 2.0f;
		m_rho0 = 0.3f;
		m_visc_cons = 0.2f;

		// m_POLY6 = 315.0f / (64.0f * M_PI * pow(m_h, 9.0f));
		m_POLY6 = 4.f / (M_PI * pow(m_h, 8.f));
		// m_SPIKY_GRAD = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_SPIKY_GRAD = -10.f / (M_PI * pow(m_h, 5.f));
		// m_VISC_LAP = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_VISC_LAP = 40.f / (M_PI * pow(m_h, 5.f));
		m_G = Eigen::Vector2d(0.f, -9.81f);

		m_grads = std::vector<Eigen::Vector2d>(m_NUM_PARTICLES, Eigen::Vector2d(0.0f));

		// ++++++++++ SPH variables +++++++++++++++++++++++

		p_density = new Grid2(m_res_x, m_res_y, m_dx);
		p_pressure = new Grid2(m_res_x, m_res_y, m_dx);
		p_divergence = new Grid2(m_res_x, m_res_y, m_dx);
		p_vorticity = new Grid2(m_res_x, m_res_y, m_dx);
		p_density->getMesh(m_renderV, m_renderF); // need to call once
		
		p_velocity = new MACGrid2(m_res_x, m_res_y, m_dx);
		p_force = new MACGrid2(m_res_x, m_res_y, m_dx);

		initSPH(0.45, 0.55, 0.7, 0.95);

		reset();
	}

	virtual void resetMembers() override {
		p_density->reset();
		particles.clear();
		initSPH(0.25, 0.75, 0.7, 0.95);
		
		// p_density->applySource(0.45, 0.55, 0.1, 0.15);
		// p_density->applySource(0.45, 0.55, 0.7, 0.95);
		// p_pressure->reset();
		// p_divergence->reset();
		// p_velocity->reset();
		// p_force->reset();
	}

	virtual void updateRenderGeometry() override {
		if (m_field == 0) {
			p_density->getColors(m_renderC);
		}
		// else if (m_field == 1) {
		// 	p_pressure->getColors(m_renderC, true);
		// }
		// else if (m_field == 2) {
		// 	p_divergence->getColors(m_renderC, true);
		// }
		// else if (m_field == 3) {
		// 	p_vorticity->getColors(m_renderC, true);
		// }
		
		// if (m_velocityOn) {
		// 	p_velocity->updateEdges(m_vScale);
		// }
	}

	virtual bool advance() override {
		// apply source in density field
		// change here to modify source for fluid
		// p_density->applySource(0.45, 0.55, 0.7, 0.95);

		// // add in new forces
		// addBuoyancy();
		// addGravity();
		// // if (m_windOn)
		// // 	addWind();

		// addForce();

		// // remove divergence
		// solvePressure();

		// // advect everything
		// advectValues();

		// // reset forces
		// p_force->reset();

		computePressureSPH();
		computeForcesSPH();
		// solveFluids();
		integrateSPH();

		// advance m_time
		m_time += m_dt;
		m_step++;

		return false;
	}

	virtual void renderRenderGeometry(
		igl::opengl::glfw::Viewer& viewer) override {
		viewer.data().set_mesh(m_renderV, m_renderF);
		viewer.data().set_colors(m_renderC);

		if (m_velocityOn) {
			viewer.data().add_edges(p_velocity->s(), p_velocity->e(), Eigen::RowVector3d(0, 0, 0));
			viewer.data().add_edges(p_velocity->vs(), p_velocity->ve(), p_velocity->vc());
		}
	}	

#pragma region FluidSteps
	void addBuoyancy() {
		double scaling = 64.0 / m_res_x;

		// add buoyancy
		for (int i = 0; i < p_force->y().size(0); ++i) {
			for (int j = 1; j < p_force->y().size(1) - 1; ++j) {
				p_force->y()(i, j) += 0.1 * (p_density->x()(i, j - 1) + p_density->x()(i, j)) / 2.0 * scaling;
			}
		}
	}

	void addGravity() {
		// add g
		for (int i = 0; i < p_force->y().size(0); ++i) {
			for (int j = 1; j < p_force->y().size(1) - 1; ++j) {
				p_force->y()(i, j) -= 9.8;
			}
		}
	}

	void addWind() {
		double scaling = 64.0 / m_res_x;

		static double r = 0.0;
		r += 1;

		const double fx = 2e-2 * cos(5e-2 * r) * cos(3e-2 * r) * scaling;

		// add wind
		for (int i = 0; i < p_force->x().size(0); ++i) {
			for (int j = 0; j < p_force->x().size(1); ++j) {
				p_force->x()(i, j) += fx;
			}
		}
	}

	void addForce() {
		for (int i = 0; i < p_velocity->x().size(0); ++i) {
			for (int j = 0; j < p_velocity->x().size(1); ++j) {
				p_velocity->x()(i, j) += m_dt * p_force->x()(i, j);
			}
		}

		for (int i = 0; i < p_velocity->y().size(0); ++i) {
			for (int j = 0; j < p_velocity->y().size(1); ++j) {
				p_velocity->y()(i, j) += m_dt * p_force->y()(i, j);
			}
		}
	}
	
	void solvePressure() {
		// copy out the boundaries 
		setNeumann();
		setZero();

		computeDivergence();

		// solve Poisson equation
		copyBorder();
		solvePoisson();

		correctVelocity();

		computeVorticity();

		// for debugging
		computeDivergence();
	}

	void setNeumann() {
		// x-velocity
		Array2d& u = p_velocity->x();
		int sx = u.size(0);
		int sy = u.size(1);
		for (int y = 0; y < sy; ++y) {
			u(0, y) = u(2, y);
			u(sx - 1, y) = u(sx - 3, y);
		}

		// y-velocity
		Array2d& v = p_velocity->y();
		sx = v.size(0);
		sy = v.size(1);
		for (int x = 0; x < sx; ++x) {
			v(x, 0) = v(x, 2);
			v(x, sy - 1) = v(x, sy - 3);
		}
	}

	void setZero() {
		// x-velocity
		Array2d& u = p_velocity->x();
		int sx = u.size(0);
		int sy = u.size(1);
		for (int x = 0; x < sx; ++x) {
			u(x, 0) = 0;
			u(x, sy - 1) = 0;
		}

		// y-velocity
		Array2d& v = p_velocity->y();
		sx = v.size(0);
		sy = v.size(1);
		for (int y = 0; y < sy; ++y) {
			v(0, y) = 0;
			v(sx - 1, y) = 0;
		}
	}

	void computeDivergence() {
		// calculate divergence
		for (int y = 1; y < m_res_y - 1; ++y) {
			for (int x = 1; x < m_res_x - 1; ++x) {
				double dudx = (p_velocity->x()(x + 1, y) - p_velocity->x()(x, y)) * m_idx;
				double dvdy = (p_velocity->y()(x, y + 1) - p_velocity->y()(x, y)) * m_idx;
				p_divergence->x()(x, y) = dudx + dvdy;
			}
		}
	}

	void computeVorticity() {
		// calculate vorticity
		for (int y = 2; y < m_res_y - 2; ++y) {
			for (int x = 2; x < m_res_x - 2; ++x) {
				double dudy = (p_velocity->x()(x, y + 1) - p_velocity->x()(x, y - 1)) * m_idx * 0.5;
				double dvdx = (p_velocity->y()(x + 1, y) - p_velocity->y()(x - 1, y)) * m_idx * 0.5;
				p_vorticity->x()(x, y) = dvdx - dudy;
			}
		}
	}

	void copyBorder() {
		Array2d& p = p_pressure->x();
		int sx = p.size(0);
		int sy = p.size(1);
		for (int y = 0; y < sy; ++y) {
			p(0, y) = p(1, y);
			p(sx - 1, y) = p(sx - 2, y);
		}
		for (int x = 0; x < sx; ++x) {
			p(x, 0) = p(x, 1);
			p(x, sy - 1) = p(x, sy - 2);
		}
	}

#pragma endregion FluidSteps

#pragma region Exercise
	void solvePoisson();
	void correctVelocity();
	void advectValues();
	void advectDensitySL(const Array2d& u, const Array2d& v);
	void advectVelocitySL(const Array2d& u, const Array2d& v);
	void MacCormackUpdate(const Array2d& d, const Array2d& d_forward, const Array2d& u,  const Array2d& u_forward, const Array2d& v, const Array2d& v_forward);
	void MacCormackClamp(const Array2d& d, const Array2d& d_forward, const Array2d& u,  const Array2d& u_forward, const Array2d& v, const Array2d& v_forward);
#pragma endregion Exercise

#pragma region SPH
	void initSPH(double xmin, double xmax, double ymin, double ymax);
	void integrateSPH();
	void computePressureSPH();
	void computeForcesSPH();
	void solveFluids();
	void solveBoundaries();
#pragma endregion SPH

#pragma region SettersAndGetters
	void selectField(int field) { m_field = field; }
	void selectVField(bool v) { m_velocityOn = v; }
	void setVelocityScale(double s) { m_vScale = s; }
	void setResX(int r) { m_res_x = r; }
	void setResY(int r) { m_res_y = r; }
	void setAccuracy(double acc) { m_acc = acc; }
	void setIteration(int iter) { m_iter = iter; }
	void setWind(bool w) { m_windOn = w; }
	void setMacCormack(bool m) { m_macOn = m; }

	//shared_ptr<ParticlesData> getParticlesData() {
	//	return m_pParticleData;
	//}

	int getField() const { return m_field; }
	bool getVField() const { return m_velocityOn; }
	double getVelocityScale() const { return m_vScale; }
	int getResX() const { return m_res_x; }
	int getResY() const { return m_res_y; }
	double getAccuracy() const { return m_acc; }
	int getIteration() const { return m_iter; }
	bool getWind() const { return m_windOn; }
	bool getMacCormack() const { return m_macOn; }
	double getTimestep() const { return m_dt; }
	Eigen::MatrixXd getVertices() const { return m_renderV;  }
	Eigen::MatrixXi getFaces() const { return m_renderF; }
#pragma endregion SettersAndGetters

private:
	int m_res_x, m_res_y;
	double m_dx, m_idx;
	double m_size_x, m_size_y;
	double m_acc;
	int m_iter;
	int m_field;
	bool m_velocityOn;
	double m_vScale;
	bool m_windOn;
	bool m_macOn;

	float m_mass;
	float m_k;
	float m_h;
	float m_rho0;
	float m_visc_cons;

	int m_NUM_PARTICLES;
	float m_POLY6;
	float m_SPIKY_GRAD;
	float m_VISC_LAP;
	Eigen::Vector2d m_G;

	std::vector<Eigen::Vector2d> m_grads;

	Grid2* p_density;
	Grid2* p_pressure;
	Grid2* p_divergence;
	Grid2* p_vorticity;
	MACGrid2* p_velocity;
	MACGrid2* p_force;

	Eigen::MatrixXd m_renderV; // vertex positions, 
	Eigen::MatrixXi m_renderF; // face indices 
	Eigen::MatrixXd m_renderC; // face (or vertex) colors for rendering

	vector<Particle> particles;
	double m_xmin;
	double m_xmax;
	double m_ymin;
	double m_ymax;

	//shared_ptr<ParticlesData> m_pParticleData;
};
