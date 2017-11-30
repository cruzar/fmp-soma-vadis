# -*- coding: utf-8 -*-
"""
Copyright (c) 2017 Pedro Miguel Miranda Queiroz Cruz

Implementam-se três métodos distintos e um método híbrido para calcular a Função Massa de
Probabilidade (FMP) da soma de Variáveis Aleatórias Discretas Independentes a partir das suas FMPs
individuais.

 - somaVADs_biv()
 - somaVADs_conv()
 - somaVADs_FFT()
 - somaVADs_hib()

Testado com:  Python version '3.6.1', Numpy version '1.13.3'
"""
import numpy as np


def somaVADs_biv(VADs):
    """ Devolve a Função Massa de Probabilidade correspondente à soma das VADs independentes cujas
        FMPs são introduzidas. A função tem aplicabilidade geral e não impõe restrições no espaço
        amostral das VADs introduzidas. [Implementado a partir da distribuição bivariada de duas
        VADs resolvida recursivamente.]

        Parâmetros
        ----------
        VADs : lista
            Cada elemento da lista deve conter um array 2D correspondente a uma VAD, contendo na
            primeira coluna os valores tomados pela variável, e na segunda coluna os respetivos
            valores da FMP.
        
        Devolve
        ----------
        par_Z_PZ : array 2D
            O formato do array de saída é igual ao formato dos arrays de entrada.
    """
    # Resolução recursiva
    
    if len(VADs) == 1:
        return VADs[0][VADs[0][:,0].argsort()]  # Ordena a abcissa e devolve
    
    else:
        X1, X2 = VADs.pop(), VADs.pop()
        
        # Somas com cada combinação de experiências aleatórias - espaco amostral da Distribuição Bivariada
        somas = np.add.outer(X1[:,0], X2[:,0])
        
        # Distribuição Bivariada
        F_W = np.multiply.outer(X1[:,1], X2[:,1])
                
        # FMP de Z - Espaço amostral em *keys* e probabilidade em *values*
        P_Z = {}
        for soma, prob in zip(somas.ravel(), F_W.ravel()):
            if soma in P_Z:
                P_Z[soma] += prob
            else:
                P_Z[soma] = prob
        
        # Costura S_Z com P_Z no mesmo array 2D
        par_Z_PZ = np.array([ list(P_Z.keys()), list(P_Z.values()) ]).T
        
        VADs.append(par_Z_PZ)
        
        return somaVADs_biv(VADs)


def somaVADs_conv(VADs, delta=1):
    """ Devolve a Função de Massa de Probabilidade em intervalos regulares correspondente à soma das
        VADs independentes cujas FMPs são introduzidas. [Implementado a partir da convolução de duas
        VADs resolvida recursivamente.]

        Parâmetros
        ----------
        VADs : lista
            Cada elemento da lista deve conter um array 2D correspondente a uma VAD, contendo na
            primeira coluna os valores tomados pela variável, e na segunda coluna os respetivos
            valores da FMP. Cada array deve estar ordenado por ordem crescente na coluna do espaço
            amostral (col 0). Além disso, o intervalo entre abcissas deve ser igual em todas as VADs.

        delta : float
            Intervalo fixo entre as abcissas de todas as VADs.
        
        Devolve
        ----------
        par_Z_PZ : array 2D
            O formato do array de saída é igual ao formato dos arrays de entrada.
    """
    # Resolução recursiva
    
    if len(VADs) == 1:
        return VADs[0]
    
    else:
        X1 = VADs.pop()
        X2 = VADs.pop()
        
        S_X1 = X1[:,0]   # Espaço amostral de X1
        S_X2 = X2[:,0]   # Espaço amostral de X2
        P_X1 = X1[:,1]   # FMP de X1
        P_X2 = X2[:,1]   # FMP de X2
        
        # Espaço amostral de Z=X1+X2
        S_Z = np.arange(S_X1[0] + S_X2[0], S_X1[-1] + S_X2[-1] + delta, delta)
        
        P_Z = np.convolve(P_X1, P_X2, mode="full")   # FMP de Z -> convolução
        
        # Costura S_Z com P_Z no mesmo array 2D
        par_Z_PZ = np.vstack([S_Z, P_Z]).T
                
        VADs.append(par_Z_PZ)
        
        return somaVADs_conv(VADs)


def somaVADs_FFT(VADs, delta=1):
    """ Devolve a Função de Massa de Probabilidade em intervalos regulares correspondente à soma das
        VADs independentes cujas FMPs são introduzidas. [Implementado a partir da convolução de
        todas as VADs passando para o domínio das frequências com FFT.]
    
        Parâmetros
        ----------
        VADs : lista de tuplos
            Introduzir uma lista de tuplos onde o primeiro elemento do tuplo seja o array 2D
            correspondente a uma VAD (contendo na primeira coluna os valores tomados pela variável,
            e na segunda coluna os respetivos valores da FMP), e o segundo elemento seja o número
            de repetições dessa VAD na soma total. Cada array deve estar ordenado por ordem
            crescente na coluna do espaço amostral (col 0). Além disso, o intervalo entre abcissas
            deve ser igual em todas as VADs.
        
        delta : float
            Intervalo fixo entre as abcissas de todas as VADs.
        
        Devolve
        ----------
        par_Z_PZ : array 2D
            O formato do array de saída é igual ao formato dos arrays de VADs de entrada.
    """
    # Lista das repetições (2os elementos de cada tuplo do input VADs)
    reps = [nested[1] for nested in VADs]
    
    # Lista dos mínimos e máximos do espaço amostral de cada VAD introduzida
    mins = [nested[0][0,0]  for nested in VADs]
    maxs = [nested[0][-1,0] for nested in VADs]
    
    # Geram-se todos os valores discretos do espaço amostral completo de
    #  Z = n_1 * VAD_1 + n_2 * VAD_2 + ... + n_i * VAD_i
    SZ_min = np.dot(reps, mins)
    SZ_max = np.dot(reps, maxs)
    S_Z = np.arange(SZ_min, SZ_max + delta, delta)
    
    # Obtem-se a Transformada de Fourier Discreta de cada VAD elevada à sua potência
    TFs = [np.fft.fft(nested[0][:,1], n=len(S_Z)) ** nested[1] for nested in VADs]
    
    # Multiplicam-se todas as T.F.
    TF = np.prod(TFs, axis=0)
    
    # Finalmente, obtem-se a Transformada Inversa da multiplicação das transformadas
    VAD_Z = np.fft.ifft(TF).real
    
    # Costura-se S_Z com P_Z no mesmo array e devolve-se
    return np.array([S_Z, VAD_Z]).T


def somaVADs_hib(VADs, delta=1):
    """ Devolve a Função de Massa de Probabilidade em intervalos regulares correspondente à soma das
        VADs independentes cujas FMPs são introduzidas. [Implementação híbrida que faz primeiro a
        convolução de todas as FMPs iguais passando ao domínio das frequências (FFT), calculando a
        potência n da transformada, e obtendo a Transformada de Fourier inversa, e depois prossegue
        fazendo convolução simples com todas as FMPs diferentes assim encontradas.]
    
        Parâmetros
        ----------
        VADs : lista de tuplos
            Introduzir uma lista de tuplos onde o primeiro elemento do tuplo seja o array 2D
            correspondente a uma VAD (contendo na primeira coluna os valores tomados pela variável,
            e na segunda coluna os respetivos valores da FMP), e o segundo elemento seja o número
            de repetições dessa VAD na soma total. Cada array deve estar ordenado por ordem
            crescente na coluna do espaço amostral (col 0).

        delta : float
            Intervalo fixo entre as abcissas de todas as VADs.
        
        Devolve
        ----------
        par_Z_PZ : array 2D
            O formato do array de saída é igual ao formato dos arrays de VADs de entrada.
    """
    ### Primeiro obtem-se a soma de todas as VADs consigo mesmas (caso haja destas somas)
    
    # Guardará lista de VADs diferentes a somar, depois de obtidas pela soma de todas as repetições
    #  da mesma VAD
    dif_VADs = []
    
    for i in range(len(VADs)):
        
        rep = VADs[i][1]  # Numero de repetições de cada VAD
        VAD = VADs[i][0]  # Array 2D com o espaço amostral e a FMP da VAD
        
        # Se só há uma repetição, adicionar à lista
        if rep == 1:
            dif_VADs.append(VAD)
        
        # Se há mais que uma repetição, fazer FFT e IFFT
        elif rep > 1:
            # Espaço amostral de Z = rep * VAD
            S_Z = np.arange(rep * VAD[0,0], (rep * VAD[-1,0] + delta), delta)
            
            # Transformada de Fourier
            sp = np.fft.fft(VAD[:,1], n=len(S_Z))
            
            # Transformada Inversa da Transformada ** rep
            FMP_final = np.fft.ifft(sp ** rep).real
            
            # Costura S_Z com FMP_final no mesmo array e insere na lista dif_VADs
            nova_VAD = np.array([S_Z, FMP_final]).T
            dif_VADs.append( nova_VAD )
            
        else:
            raise ValueError('O número de repetições deve ser igual ou superior a 1.')
    
    ### Agora, tendo já a lista sem VADs repetidas, fazemos a convolução destas FMPs.
    return somaVADs_conv( dif_VADs )
