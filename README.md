# Função Massa de Probabilidade da soma de Variáveis Aleatórias Discretas Independentes
Implementam-se três métodos distintos e um método híbrido para calcular a Função Massa de Probabilidade (FMP) da soma de Variáveis Aleatórias Discretas Independentes a partir das suas FMPs individuais.

Um dos métodos é construído a partir da distribuição bivariada de duas VADs e resolvido recursivamente para *n* VADs; outro método implementa a convolução das FMPs de duas VADs, resolvida recursivamente para *n* VADs; um terceiro método implementa a convolução das FMPs no domínio das frequências utilizando FFT; e um quarto método mistura a convolução no domínio direto com a convolução no domínio das frequências.

__________________
up200506513@fc.up.pt
