#include <igl/writeOFF.h>
#include <thread>
#include "Gui.h"
#include "Simulator.h"
#include "FluidSim.h"

/*
 * GUI for the spring simulation. This time we need additional paramters,
 * e.g. which integrator to use for the simulation and the force applied to the
 * canonball, and we also add some more visualizations (trajectories).
 */
class FluidGui : public Gui {
public:
	float m_dt;
	float m_acc;
	int m_iter;
	float m_vScale;
	bool m_windOn;
	bool m_macOn;

	const vector<char const*> m_fields = {
	   "Density", "Pressure", "Divergence", "Vorticity"
	};
	int m_selected_field;
	bool m_velocityOn;

	FluidSim* p_fluidSim = NULL;

	FluidGui() {
		turnOffLight(); // no light for field visualization
		orthoCam(); // orthographic camera

		p_fluidSim = new FluidSim();
		m_dt = p_fluidSim->getTimestep();
		m_acc = p_fluidSim->getAccuracy();
		m_iter = p_fluidSim->getIteration();
		m_vScale = p_fluidSim->getVelocityScale();
		m_windOn = p_fluidSim->getWind();
		m_macOn = p_fluidSim->getMacCormack();
		m_selected_field = p_fluidSim->getField();
		m_velocityOn = p_fluidSim->getVField();
		setSimulation(p_fluidSim);
		setFastForward(true);
		//setParticleSystem(p_fluidSim->getParticlesData());
		start();
	}

	virtual void updateSimulationParameters() override {
		// change all parameters of the simulation to the values that are set in
		// the GUI
		p_fluidSim->setVelocityScale(m_vScale);
		p_fluidSim->setTimestep(m_dt);
		p_fluidSim->setAccuracy(m_acc);
		p_fluidSim->setIteration(m_iter);
	}

	virtual void clearSimulation() override {
		p_fluidSim->setWind(m_windOn);
		p_fluidSim->setMacCormack(m_macOn);
	}

	virtual void drawSimulationParameterMenu() override {
		if (ImGui::Combo("Fields", &m_selected_field,
			m_fields.data(), m_fields.size())) {
			cout << m_selected_field << endl;
			p_fluidSim->selectField(m_selected_field);
			p_fluidSim->updateRenderGeometry();
		}
		if (ImGui::Checkbox("Velocity", &m_velocityOn)) {
			p_fluidSim->selectVField(m_velocityOn);
			p_fluidSim->updateRenderGeometry();
		}
		ImGui::InputFloat("v scale", &m_vScale, 0, 0);
		ImGui::InputFloat("dt", &m_dt, 0, 0);
		ImGui::InputFloat("accuracy", &m_acc, 0, 0, 5);
		ImGui::InputInt("iter", &m_iter, 0, 0);
		if (ImGui::Checkbox("Wind", &m_windOn))
			p_fluidSim->setWind(m_windOn);
		if (ImGui::Checkbox("MacCormack", &m_macOn))
			p_fluidSim->setMacCormack(m_macOn);
	}
};

int main(int argc, char* argv[]) {
	// create a new instance of the GUI for the spring simulation
	new FluidGui();

	return 0;
}