import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

umidade = ctrl.Antecedent(np.arange(0, 101, 1), 'umidade')
npessoas = ctrl.Antecedent(np.arange(0, 26, 1), 'npessoas')
umidificador = ctrl.Antecedent(np.arange(0, 101, 1), 'umidificador')

umidade.automf(3)
umidade.automf(3)

umidificador['alto'] = fuzz.trimf(umidificador.universe, [0, 0, 50])
umidificador['normal'] = fuzz.trimf(umidificador.universe, [0, 50, 100])
umidificador['desligado'] = fuzz.trimf(umidificador.universe, [50, 100, 100])

umidade.view()
plt.show()