import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#  Universo de variaveis.
x_umidade = np.arange(0, 101, 1)
x_npessoas = np.arange(0, 21, 1)
x_umidificador = np.arange(0, 101, 1)


#   Funções fuzzy
umid_ruim = fuzz.trapmf(x_umidade, [0, 0, 40, 50])
umid_boa = fuzz.trimf(x_umidade, [40, 60, 80])
umid_excessiva = fuzz.trimf(x_umidade, [70, 100, 100])

npessoas_nenhuma = fuzz.trapmf(x_npessoas, [0, 0, 0, 0])
npessoas_poucas = fuzz.trapmf(x_npessoas, [1, 1, 5, 12])
npessoas_muitas = fuzz.trapmf(x_npessoas, [6, 15, 21, 21])

umidif_alto = fuzz.trapmf(x_umidificador, [0, 0, 25, 50])
umidif_normal = fuzz.trimf(x_umidificador, [20, 50, 80])
umidif_desligado = fuzz.trapmf(x_umidificador, [50, 75, 100, 100])


fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_umidade, umid_ruim, 'r', label='ruim')
ax0.plot(x_umidade, umid_boa, 'g', label='boa')
ax0.plot(x_umidade, umid_excessiva, 'b', label='excessiva')
ax0.set_title('Qualidade da Umidade do Ar')
#ax0.set_xlabel('Umidade do ar')
ax0.legend()

ax1.plot(x_npessoas, npessoas_nenhuma, 'r', label='nenhuma')
ax1.plot(x_npessoas, npessoas_poucas, 'g', label='poucas')
ax1.plot(x_npessoas, npessoas_muitas, 'b', label='muitas')
ax1.set_title('Numero de Pessoas')
ax1.legend()

ax2.plot(x_umidificador, umidif_alto, 'r', label='alto')
ax2.plot(x_umidificador, umidif_normal, 'g', label='normal')
ax2.plot(x_umidificador, umidif_desligado, 'b', label='desligado')
ax2.set_title('Comportamento do Umidificador')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
#plt.show()

#Regras

valor_umidade = 70
valor_pessoas = 13

qualid_umidade_ruim = fuzz.interp_membership(x_umidade, umid_ruim, valor_umidade)
qualid_umidade_boa = fuzz.interp_membership(x_umidade, umid_boa, valor_umidade)
qualid_umidade_exces = fuzz.interp_membership(x_umidade, umid_excessiva, valor_umidade)

numero_pess_nenhuma = fuzz.interp_membership(x_npessoas, npessoas_nenhuma, valor_pessoas)
numero_pess_poucas = fuzz.interp_membership(x_npessoas, npessoas_poucas, valor_pessoas)
numero_pess_muitas = fuzz.interp_membership(x_npessoas, npessoas_muitas, valor_pessoas)


#Regra 1
ativar_regra1 = np.fmax(qualid_umidade_exces, numero_pess_nenhuma)
ativa_umidificado_desligado = np.fmin(ativar_regra1, umidif_desligado)

#Regra 2
ativar_regra2 = np.fmax(qualid_umidade_boa, numero_pess_poucas)
ativa_umidificado_normal = np.fmin(ativar_regra2, umidif_normal)

#Regra 3
ativar_regra3 = np.fmax(qualid_umidade_ruim, numero_pess_muitas)
ativa_umidificado_alto = np.fmin(ativar_regra3, umidif_alto)

umidificador0 = np.zeros_like(x_umidificador)

#   Visualização
fig1, ax4 = plt.subplots(figsize=(8, 3))

ax4.fill_between(x_umidificador, umidificador0, ativa_umidificado_desligado, facecolor='b', alpha=0.7, label='desligado')
ax4.plot(x_umidificador, umidif_desligado, 'b', linewidth=0.5, linestyle='--', )
ax4.fill_between(x_umidificador, umidificador0, ativa_umidificado_normal, facecolor='g', alpha=0.7, label='normal')
ax4.plot(x_umidificador, umidif_normal, 'g', linewidth=0.5, linestyle='--')
ax4.fill_between(x_umidificador, umidificador0, ativa_umidificado_alto, facecolor='r', alpha=0.7, label='alto')
ax4.plot(x_umidificador, umidif_alto, 'r', linewidth=0.5, linestyle='--')
ax4.set_title('Comportamento do Umidificador')
ax4.legend()

for ax in (ax4,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# Aggregate all three output membership functions together
aggregated = np.fmax(ativa_umidificado_alto,
                     np.fmax(ativa_umidificado_normal, ativa_umidificado_desligado))

# Calculate defuzzified result
tip = fuzz.defuzz(x_umidificador, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(x_umidificador, aggregated, tip)  # for plot

# Visualize this
fig3, ax5 = plt.subplots(figsize=(8, 3))

ax5.plot(x_umidificador, umidif_alto, 'b', linewidth=0.5, linestyle='--', )
ax5.plot(x_umidificador, umidif_normal, 'g', linewidth=0.5, linestyle='--')
ax5.plot(x_umidificador, umidif_desligado, 'r', linewidth=0.5, linestyle='--')
ax5.fill_between(x_umidificador, umidificador0, aggregated, facecolor='Orange', alpha=0.7)
ax5.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9, label=f'Saída = {tip:.2f}')
ax5.set_title('Associação e resultado agregado (linha)')
ax5.legend()

for ax in (ax5,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

plt.show()
