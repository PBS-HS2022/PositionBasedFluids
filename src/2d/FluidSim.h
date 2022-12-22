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
		m_iter = 10;
		m_field = 0;
		m_velocityOn = false;
		m_vScale = 20;
		m_windOn = false;
		m_macOn = true;

		// ++++++++++ SPH / PBD variables +++++++++++++++++++++++
		m_NUM_PARTICLES = 2500;

		m_mass = 2.5f;
		m_k = 100.0f;
		m_h = 4.f;
		m_rho0 = 0.3f;
		m_visc_cons = 0.f;

		m_eta = 0.01f; // 1%
		m_min_iterations = 3; // idfk
		m_rho_err = std::vector<float>(m_NUM_PARTICLES);

		// m_POLY6 = 315.0f / (64.0f * M_PI * pow(m_h, 9.0f));
		m_POLY6 = 4.f / (M_PI * pow(m_h, 8.f));
		// m_SPIKY_GRAD = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_SPIKY_GRAD = -10.f / (M_PI * pow(m_h, 5.f));
		// m_VISC_LAP = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_VISC_LAP = 40.f / (M_PI * pow(m_h, 5.f));
		m_G = Eigen::Vector2d(0.f, -9.81f);

		m_grads = std::vector<Eigen::Vector2d>(m_NUM_PARTICLES, Eigen::Vector2d(0.0f, 0.0f));

		// ++++++++++ SPH / PBD variables +++++++++++++++++++++++

		p_density = new Grid2(m_res_x, m_res_y, m_dx);
		p_density->getMesh(m_renderV, m_renderF); // need to call once

		initSPH(0.45, 0.55, 0.7, 0.95);

		reset();
	}

	virtual void resetMembers() override {
		p_density->reset();
		particles.clear();
		initSPH(0.25, 0.75, 0.7, 0.95);
	}

	virtual void updateRenderGeometry() override {
		if (m_field == 0) {
			p_density->getColors(m_renderC);
		}
	}

	virtual bool advance() override {
		// Call the SPH / PBD integration
		integrateSPH();

		// advance m_time
		m_time += m_dt;
		m_step++;

		return false;
	}

	// Used for PCISPH attempt (not used for PBD anymore)
	bool predict() {
		computePressureSPH();
		computeForcesSPH();
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

	virtual void exportObj() override {
	}

#pragma region SPH
	void initSPH(double xmin, double xmax, double ymin, double ymax);
	std::vector<std::vector<int>> findNeighbors();
	void integrateSPH();
	void computePressureSPH();
	void computePressurePCISPH();
	void computeForcesSPH();
	void solveFluids(std::vector<std::vector<int>> * neighbors);
	void solveBoundaries();
	void applyViscosity(std::vector<std::vector<int>> * neighbors, int i);
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

	void setNumParticles(int num) { m_NUM_PARTICLES = num; }
	void setKernelRadius(float h) { m_h = h; }
	void setVisc(float v) { m_visc_cons = v; }
	void setRestDensity(float rho) { m_rho0 = rho; }

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

	int getNumParticles() const { return m_NUM_PARTICLES; }
	float getKernelRadius() const { return m_h; }
	float getVisc() const { return m_visc_cons; }
	float getRestDensity() const { return m_rho0; }

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

	float m_eta; // Density converging threshold for Incompressibility
	int m_min_iterations; // Number of iterations before considering a convergeance
	std::vector<float> m_rho_err; // For density prediction

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
