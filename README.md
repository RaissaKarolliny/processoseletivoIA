# Processo Seletivo – Intensivo Maker | AI
## 📝 Relatório do Candidato

👤 Identificação: Raissa Karoliny da Silva Rodrigues

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura CNN implementada foi projetada com profundidade moderada, focando em Edge AI para garantir baixa latência e eficiência computacional.
- **Extração de Características:** Duas camadas convolucionais (`Conv2D` com 16 e 32 filtros de 3x3) seguidas de `MaxPooling2D`. A primeira foca em features de baixo nível (bordas) e a segunda em padrões mais complexos.
- **Classificação:** Após o achatamento (`Flatten`), uma camada `Dense` (64 neurônios) agrega as características espaciais, finalizando na saída `Softmax` (10 neurônios).

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow / Keras:** Construção, treinamento, conversão (TFLite) e extração base da Matriz de Confusão.
- **NumPy:** Manipulação matricial, pré-processamento e cálculo manual das métricas macro-average.

### 3️⃣ Técnica de Otimização do Modelo

Para viabilizar a execução em dispositivos embarcados, foi realizada a conversão do modelo para o formato TensorFlow Lite, aplicando técnicas de quantização.

Inicialmente, foi utilizada a Dynamic Range Quantization, na qual os pesos do modelo são convertidos de float32 para representações inteiras de menor precisão (tipicamente int8). Essa abordagem reduz significativamente o tamanho do modelo e melhora a eficiência de execução, especialmente em ambientes com CPU e recursos limitados.

Adicionalmente, foi explorada a Float16 Quantization, que converte os pesos para float16, reduzindo o uso de memória com menor impacto potencial na precisão.

### 4️⃣ Resultados Obtidos
As métricas calculadas em formato Macro-average demonstram:
- Alta Acurácia no conjunto de Teste.
- Elevados valores de **Precision**, **Recall** e **F1-score (Macro)**, indicando que o modelo tem pouca confusão entre as classes e aprendeu de forma equilibrada a distinguir todos os 10 dígitos.

| Métrica                  | Valor  |
|--------------------------|--------|
| Acurácia de Treino       | 0.9879 |
| Acurácia de Validação    | 0.9888 |
| Acurácia de Teste        | 0.9862 |
| Precision (Macro)        | 0.9862 |
| Recall (Macro)           | 0.9862 |
| F1-score (Macro)         | 0.9861 |
| Perda (Loss)             | 0.0456 |

Após a conversão para TensorFlow Lite, foi realizada a comparação entre técnicas de quantização:

| Técnica de Quantização    | Tamanho |
|--------------------------|--------|
| Dynamic Range            | 63.54 KB |
| Float16                 | 116.36 KB |

Observa-se que a Dynamic Range Quantization apresentou melhor eficiência de compressão, reduzindo significativamente o tamanho do modelo.

Além disso, essa técnica é mais adequada para execução em CPU, pois não depende de suporte específico de hardware para operações em ponto flutuante de menor precisão.

Dessa forma, a Dynamic Range Quantization foi escolhida como solução final, por oferecer o melhor equilíbrio entre desempenho, tamanho do modelo e compatibilidade com sistemas embarcados.

### 5️⃣ Comentários Adicionais

- **Decisões técnicas importantes:** Escolha de Hiperparâmetros (Treinamento)

Foram testadas duas configurações de treinamento:

| Configuração | Acurácia | Perda (Loss) |
|--------------|---------|-------------|
| A (3 épocas, batch 16)  | 0.9886  | 0.0344 |
| B (5 épocas, batch 128) | 0.9858  | 0.0425 |

A Configuração A apresentou melhor desempenho em todas as métricas (acurácia, precisão, recall e F1), além de menor perda. 

Do ponto de vista de Edge AI, a Configuração A foi escolhida porque:

Menor custo computacional (menos épocas e batches menores → menor uso de memória/CPU);
Treinamento mais rápido, compatível com ambientes restritos (CI/CPU);
Maior robustez à quantização: modelos com melhor desempenho inicial tendem a sofrer menor degradação após conversão para TensorFlow Lite;
Estabilidade: batches menores tendem a produzir atualizações mais estáveis em cenários com recursos limitados.

Para mais informações entre em contato: raissateixeir4@gmail.com
