#include <igl/edges.h>
#include "Simulation.h"
#include "Grid3.h"
#include <igl/writeOBJ.h>

using namespace std;

// Struct for water particles
// Set initial position, velocity, force, density, pressure
struct Particle {
	Particle(float _x, float _y, float _z)
		: x(_x, _y, _z), v(0.0f, 0.0f, 0.0f), f(0.0f, 0.0f, 0.0f), rho(0), p(0.0f) {}

	Eigen::Vector3d x, v, f;
	float rho, p;
};

/*
 * Simulation of a simple smoke plume rising.
 */
class FluidSim : public Simulation {
public:
	FluidSim() : Simulation() { init(); }

	virtual void init() override {
		m_res_x = 80;
		m_res_y = int(m_res_x*1.5); // 3:2 ratio
		m_res_z = 80;
		m_size_x = m_res_x; // or just 1.0
		m_dx = m_size_x / m_res_x; // ! dx == dy
		m_idx = m_res_x / m_size_x;
		m_size_y = m_dx * m_res_y;
		m_dt = 0.005 * sqrt((m_res_x + m_res_y) * 0.5); // TODO: does this need to account for z?
		// m_dt = 0.01;
		m_acc = 1e-5;
		m_iter = 10;
		m_field = 0;
		m_velocityOn = false;
		m_vScale = 20;
		m_windOn = false;
		m_macOn = true;

		m_show3dPoints = false;

		// ++++++++++ SPH variables +++++++++++++++++++++++
		m_NUM_PARTICLES = 250000;

		m_mass = 2.5f;
		m_k = 100.0f;
		m_h = 0.2f;
		m_rho0 = 0.0001f;
		m_visc_cons = 0.f;

		// TODO: do these need to account for z?
		// m_POLY6 = 315.0f / (64.0f * M_PI * pow(m_h, 9.0f));
		m_POLY6 = 4.f / (M_PI * pow(m_h, 8.f));
		// m_SPIKY_GRAD = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_SPIKY_GRAD = -10.f / (M_PI * pow(m_h, 5.f));
		// m_VISC_LAP = 45.0f / (M_PI * pow(m_h, 6.0f));
		m_VISC_LAP = 40.f / (M_PI * pow(m_h, 5.f));
		m_G = Eigen::Vector3d(0.f, -9.81f, 0.f);

		m_grads = std::vector<Eigen::Vector3d>(m_NUM_PARTICLES, Eigen::Vector3d(0.0f, 0.0f, 0.0f));

		// ++++++++++ SPH variables +++++++++++++++++++++++

		p_density = new Grid3(m_res_x, m_res_y, m_res_z, m_dx, true);
		p_pressure = new Grid3(m_res_x, m_res_y, m_res_z, m_dx);
		p_divergence = new Grid3(m_res_x, m_res_y, m_res_z, m_dx);
		p_vorticity = new Grid3(m_res_x, m_res_y, m_res_z, m_dx);

		// reset() calls resetMembers() after setting timestamp to 0.
		reset();

		// updateRenderGeometry() is called after this by the Gui
		// (in Gui::setSimulation), which will create the first frame's mesh.
	}

	virtual void resetMembers() override {
		p_density->reset();
		particles.clear();
		initSPH(0.45, 0.55, 0.7, 0.95, 0.45, 0.55);
	}

	virtual void updateRenderGeometry() override {
		// Build the mesh at every timestep, since unlike the 2D variant where
		// the mesh is always a constant grid (colored black or blue), 3D fluid
		// meshes are constructed depending on the density at every frame.
		if (!m_show3dPoints) {
			p_density->buildMesh();
			// Put the generated mesh into our local variables.
			p_density->getMesh(m_renderV, m_renderF);
		}

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
		
	}

	virtual bool advance() override {
		// computePressureSPH();
		// computeForcesSPH();
		// solveFluids();
		integrateSPH();

		// advance m_time
		m_time += m_dt;
		m_step++;

		return false;
	}

	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer& viewer) override {
		if (!m_show3dPoints) {
			viewer.data().set_mesh(m_renderV, m_renderF);
			viewer.data().set_colors(m_renderC);
		} else {
			Eigen::MatrixXd particlePos(particles.size(), 3);
			for (int i = 0; i < particles.size(); i++) {
				particlePos.row(i) = particles[i].x - Eigen::Vector3d(m_res_x / 2., 0, m_res_z / 2.);
			}
			viewer.data().point_size = 20.f;
			viewer.data().add_points(particlePos, m_renderC);
		}
 	}

	virtual void exportObj() override {
		// We write to a sequence of object files that can be loaded in Blender
		// I had no idea how to do an sprintf-like thing in c++, so here's some
		// stream magic from stackoverflow
		std::ostringstream filenameStream;
  		filenameStream << "obj_frames/output" << m_step << ".obj";
		igl::writeOBJ(filenameStream.str(), m_renderV, m_renderF);
	}

#pragma region SPH
	void initSPH(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
	std::vector<std::vector<int>> findNeighbors();
	void integrateSPH();
	void computePressureSPH();
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

	void setShow3dPoints(bool b) { m_show3dPoints = b; }

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

	bool getShow3dPoints() const { return m_show3dPoints; }

	double getTimestep() const { return m_dt; }

	int getNumParticles() const { return m_NUM_PARTICLES; }
	float getKernelRadius() const { return m_h; }
	float getVisc() const { return m_visc_cons; }
	float getRestDensity() const { return m_rho0; }

	Eigen::MatrixXd getVertices() const { return m_renderV;  }
	Eigen::MatrixXi getFaces() const { return m_renderF; }
#pragma endregion SettersAndGetters

private:
	int m_res_x, m_res_y, m_res_z;
	double m_dx, m_idx;
	double m_size_x, m_size_y;
	double m_acc;
	int m_iter;
	int m_field;
	bool m_velocityOn;
	double m_vScale;
	bool m_windOn;
	bool m_macOn;
	
	bool m_show3dPoints;

	float m_mass;
	float m_k;
	float m_h;
	float m_rho0;
	float m_visc_cons;

	int m_NUM_PARTICLES;
	float m_POLY6;
	float m_SPIKY_GRAD;
	float m_VISC_LAP;
	Eigen::Vector3d m_G;

	std::vector<Eigen::Vector3d> m_grads;

	Grid3* p_density;
	Grid3* p_pressure;
	Grid3* p_divergence;
	Grid3* p_vorticity;

	Eigen::MatrixXd m_renderV; // vertex positions, 
	Eigen::MatrixXi m_renderF; // face indices 
	Eigen::MatrixXd m_renderC; // face (or vertex) colors for rendering

	vector<Particle> particles;
	double m_xmin;
	double m_xmax;
	double m_ymin;
	double m_ymax;
	double m_zmin;
	double m_zmax;

	//shared_ptr<ParticlesData> m_pParticleData;
};
