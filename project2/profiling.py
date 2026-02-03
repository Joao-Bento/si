import os
import pandas as pd
import numpy as np

csv_path = '/home/joao/si/project2/profile.csv'

profile = pd.read_csv(csv_path)

print(profile.head())

profile = profile.iloc[2:]

profile['Género'] = profile['Género'].replace({'Masculino': 'Male', 'Feminino': 'Female'})
profile['Mão dominante'] = profile['Mão dominante'].replace({'Direita': 'Right', 'Esquerda': 'Left'})
profile['Nível de familiaridade com realidade aumentada'] = profile['Nível de familiaridade com realidade aumentada'].replace({'1 (primeira vez)': '1'})
profile['Utilizou alguma forma de correção ocular (óculos, lentes de contacto, etc.) durante o exercício?'] = profile['Utilizou alguma forma de correção ocular (óculos, lentes de contacto, etc.) durante o exercício?'].replace({'Sim': 'Yes', 'Não': 'No'})

print(profile.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
age_counts = profile['Idade'].value_counts()
plt.bar(age_counts.index, age_counts.values, color='blue')
plt.xlabel('Age')
plt.ylabel('Number of volunteers')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('/home/joao/si/project2/profile_idade_bar_chart.png')
plt.show()

plt.figure(figsize=(4, 4))
gender_counts = profile['Género'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['orange', 'red'])
#plt.title('Gender Distribution')
plt.savefig('/home/joao/si/project2/profile_genero_pie_chart.png')
plt.show()

plt.figure(figsize=(4, 4))
handedness_counts = profile['Mão dominante'].value_counts()
plt.pie(handedness_counts.values, labels=handedness_counts.index, autopct='%1.1f%%', startangle=90, colors=['orange', 'red'])
#plt.title('Dominant Hand Distribution')
plt.savefig('/home/joao/si/project2/profile_mao_dominante_pie_chart.png')
plt.show()

plt.figure(figsize=(4, 4))
familiarity_counts = profile['Nível de familiaridade com realidade aumentada'].value_counts().sort_index()
plt.pie(familiarity_counts.values, labels=familiarity_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, len(familiarity_counts))))
#plt.title('Familiarity with Augmented Reality')
plt.savefig('/home/joao/si/project2/profile_familiarity_pie_chart.png')
plt.show()

plt.figure(figsize=(4, 4))
correction_counts = profile['Utilizou alguma forma de correção ocular (óculos, lentes de contacto, etc.) durante o exercício?'].value_counts()
plt.pie(correction_counts.values, labels=correction_counts.index, autopct='%1.1f%%', startangle=90, colors=['orange', 'red'])
#plt.title('Use of Vision Correction During Exercise')
plt.savefig('/home/joao/si/project2/profile_vision_correction_pie_chart.png')
plt.show()
