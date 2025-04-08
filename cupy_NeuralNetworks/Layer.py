import numpy as np
import cupy as cp
import copy
import pickle

class Layer:
    def __init__(self, output_shape, input_shape, activation):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.activation = activation
        self.build = False

    def built(self): pass
    def forward(self, X): pass
    def backward(self, dL_A, optimizer, lr): pass
    def getOutput(self): pass
    def getName(self): pass
    def save(self): pass
    def load(self): pass

class Dense(Layer):
    def __init__(self, output_shape, input_shape=0, activation=None, **kwargs):
        super().__init__(output_shape, input_shape, activation)
        self.W = None
        self.b = None

    def built(self):
        if self.build == False:
            self.W = cp.random.randn(self.input_shape, self.output_shape)
            self.b = cp.random.randn(1, self.output_shape)
            self.grad_W = cp.zeros_like(self.W)
            self.grad_b = cp.zeros_like(self.b)
            self.build = True

    def forward(self, X):
        self.built()
        self.X = X
        self.Z = cp.dot(X, self.W) + self.b
        self.A = self.activation(self.Z)

        return self.A
    
    def backward(self, dL_A, optimizer, lr):
        self.dA_Z = self.activation.deri
        self.dZ_W = self.X
        self.dZ_X = self.W

        if self.dA_Z.ndim == 3: dL_Z = (cp.expand_dims(dL_A, axis=1) * self.dA_Z).sum(axis=0)
        else: dL_Z = dL_A * self.dA_Z
        dL_W = cp.dot(dL_Z, self.dZ_W)
        dL_b = cp.sum(dL_Z, axis=1, keepdims=True)

        self.grad_W, self.grad_b = optimizer(dL_W, dL_b)
        assert self.grad_W.shape[0] == self.W.shape[0] and self.grad_W.shape[1] == self.W.shape[1]
        self.W -= lr*self.grad_W
        self.b -= lr*self.grad_b

        return cp.dot(self.dZ_X, dL_Z)
        
    def getOutput(self):
        return self.A

    def getName(self):
        return 'Dense' +'('+ self.activation.getName() + ')'
    
    def save(self):
        cp.save("weight.npy", self.W)
        cp.save("bias.npy", self.b)

    def load(self):
        self.b = cp.load('bias.npy')
        self.W = cp.load('weight.npy')
    
    def __call__(self, X):
        return self.forward(X)
        return self.getOutput()


#----------------------------------------------------------------------------------
# Hidden Layer = RNN cell (Tensorflow)
# output_shape = nums hidden units = number of neurons in Hidden Layer
# input_shape = (timesteps, num_features)
# Input X shape must be (batch_size, timesteps, num_features)
# U = weight from Input to Hidden Layer
# W = weight from Hidden to Hidden
# V = weight from Hidden to Output 
# k = truncated step

class RNN(Layer):
    def __init__(self, output_shape, input_shape=0, activation=None, initial_state=None, return_sequences=False, **kwargs):
        super().__init__(output_shape, input_shape, activation)
        self.return_sequences = return_sequences
        self.initial_state = initial_state
        self.kwargs = kwargs
        self.U = None
        self.W = None
        self.bU = None

    def built(self):
        if self.build == False:
            self.U = cp.random.randn(self.input_shape[1], self.output_shape)
            self.W = cp.random.randn(self.output_shape, self.output_shape)
            self.bU = cp.random.randn(1, self.output_shape)
            self.grad_U = cp.zeros_like(self.U)
            self.grad_W = cp.zeros_like(self.W)
            self.grad_bU = cp.zeros_like(self.b)
            self.build = True        

    def forward(self, X):
        self.built()
        self.X = X
        self.H = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        if self.initial_state != None: self.H[0] = self.initial_state
        self.hidden_activation = [0]

        for t in range(1,self.input_shape[0]+1):
            Zx = cp.dot(X[:,t-1,:], self.U) + self.bU
            Zh = cp.dot(self.H[t-1], self.W)
            Z = Zx + Zh
            self.hidden_activation.append(copy.deepcopy(self.activation))
            self.H[t] = self.hidden_activation[t](Z)

        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def backward(self, dL_A, optimizer, lr):
        k = self.kwargs['truncated_step']

        if dL_A.ndim == 2:
            dL_H = dL_A
            dL_U = 0
            dL_W = 0
            dL_bU = 0

            for t in reversed(range(self.input_shape[0] + 1 - k, self.input_shape[0] + 1)):
                dH_Z = self.hidden_activation[t].deri
                dZ_U = self.X[:,t-1,:]
                dZ_W = self.H[t-1]
                dZ_H = self.W

                dL_Z = dL_H * dH_Z
                dL_U += cp.dot(dL_Z, dZ_U)
                # self.grad_U += dL_U.T
                dL_W += cp.dot(dL_Z, dZ_W)
                # self.grad_W += dL_W.T
                dL_bU += cp.sum(dL_Z, axis=1, keepdims=True)
                dL_H = cp.dot(dZ_H, dL_Z)

            self.grad_U, self.grad_W, self.grad_bU = optimizer(dL_U, dL_W, dL_bU)
            self.U -= lr*self.grad_U
            self.W -= lr*self.grad_W
            self.bU -= lr*self.grad_bU

        else:
            pass


    def getOutput(self):
        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def getName(self):
        return 'RNN' +'('+ self.activation.getName() + ')'
    
    def __call__(self, X):
        return self.forward(X)
    
#----------------------------------------------------------------------------------

class LSTM(Layer):
    def __init__(self, output_shape, input_shape=0, activation=None, recurrent_activation=None, initial_state=None, return_sequences=False, **kwargs):
        super().__init__(output_shape, input_shape, activation)
        self.recurrent_activation = recurrent_activation
        self.return_sequences = return_sequences
        self.initial_state = initial_state
        self.kwargs = kwargs
        self.Ui = None
        self.Wi = None
        self.Uo = None
        self.Wo = None
        self.Uc = None
        self.Wc = None
        self.bi = None
        self.bo = None
        self.bc = None

    def built(self):
        if self.build == False:
            self.Ui = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16) 
            self.Wi = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16) 
            self.Uo = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16) 
            self.Wo = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16) 
            self.Uc = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16) 
            self.Wc = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16) 
            self.bi = cp.random.randn(1, self.output_shape) * 0.2
            self.bo = cp.random.randn(1, self.output_shape) * 0.2
            self.bc = cp.random.randn(1, self.output_shape) * 0.2
            self.grad_Ui = cp.zeros_like(self.Ui)
            self.grad_Wi = cp.zeros_like(self.Wi)
            self.grad_Uo = cp.zeros_like(self.Uo)
            self.grad_Wo = cp.zeros_like(self.Wo)
            self.grad_Uc = cp.zeros_like(self.Uc)
            self.grad_Wc = cp.zeros_like(self.Wc)
            self.grad_bi = cp.zeros_like(self.bi)
            self.grad_bo = cp.zeros_like(self.bo)
            self.grad_bc = cp.zeros_like(self.bc)
            self.build = True
    

    def forward(self, X):
        # I = input gate, O = output gate, C = cell state, H = hidden state
        # Zi = X.Ui + H.Wi + bi
        # I = recurrent_activation(Zi)
        # Zo = X.Uo + H.Wo + bo
        # O = recurrent_activation(Zo)
        # Zc = X.Uc + H.Wc + bc
        # Ctile = activation(Zc)
        # C = np + I*Ctile
        # H = O*activation(C)

        self.built()
        self.X = X
        self.I = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.O = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.Ctile = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.C = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.H = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        if self.initial_state != None: self.H[0], self.C[0] = self.initial_state
        self.hidden_activation = [0]
        self.unit_activation = {'i': [0], 'o':[0], 'c':[0]}

        for t in range(1,self.input_shape[0]+1):
            self.unit_activation['i'].append(copy.deepcopy(self.recurrent_activation))
            self.I[t] = self.unit_activation['i'][t](cp.dot(X[:,t-1,:], self.Ui) + cp.dot(self.H[t-1], self.Wi) + self.bi)
            self.unit_activation['o'].append(copy.deepcopy(self.recurrent_activation))
            self.O[t] = self.unit_activation['o'][t](cp.dot(X[:,t-1,:], self.Uo) + cp.dot(self.H[t-1], self.Wo) + self.bo)
            self.unit_activation['c'].append(copy.deepcopy(self.activation))
            self.Ctile[t] = self.unit_activation['c'][t](cp.dot(X[:,t-1,:], self.Uc) + cp.dot(self.H[t-1], self.Wc) + self.bc)
            self.C[t] = self.C[t-1] + self.I[t] * self.Ctile[t]
            self.hidden_activation.append(copy.deepcopy(self.activation))
            self.H[t] = self.O[t] *  self.hidden_activation[t](self.C[t])

        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def backward(self, dL_A, optimizer, lr):
        # df_C: derivative of activation function wrt C

        k = self.kwargs['truncated_step']

        if dL_A.ndim == 2:
            dL_H = dL_A
            dL_C = 0
            dL_Ui = dL_Uo = dL_Uc = 0
            dL_Wi = dL_Wo = dL_Wc = 0
            dL_bi = dL_bo = dL_bc = 0

            for t in reversed(range(self.input_shape[0] + 1 - k, self.input_shape[0] + 1)):
                dZi_Ui = dZo_Uo = dZc_Uc = self.X[:,t-1,:]
                dZi_Wi = dZo_Wo = dZc_Wc = self.H[t-1]
                dZi_H = self.Wi
                dZo_H = self.Wo
                dZc_H = self.Wc
                dC_np = cp.ones_like(self.C[t-1]).T
                dC_I = self.Ctile[t].T
                dC_Ctile = self.I[t].T
                df_C = self.hidden_activation[t].deri
                dH_f = self.O[t].T
                dH_O = self.hidden_activation[t].a.T
                

                dL_C = dL_H * dH_f * df_C + dL_C*dC_np
                dL_Zi = (dL_C) * dC_I * self.unit_activation['i'][t].deri
                dL_Zo = (dL_H * dH_O) * self.unit_activation['o'][t].deri 
                dL_Zc = (dL_C) * dC_Ctile * self.unit_activation['c'][t].deri
                    
                dL_Ui += cp.dot(dL_Zi, dZi_Ui) 
                dL_Wi += cp.dot(dL_Zi, dZi_Wi) 
                dL_Uo += cp.dot(dL_Zo, dZo_Uo)    
                dL_Wo += cp.dot(dL_Zo, dZo_Wo)
                dL_Uc += cp.dot(dL_Zc, dZc_Uc)
                dL_Wc += cp.dot(dL_Zc, dZc_Wc)
                dL_bi += cp.sum(dL_Zi, axis=1, keepdims=True)
                dL_bo += cp.sum(dL_Zo, axis=1, keepdims=True)
                dL_bc += cp.sum(dL_Zc, axis=1, keepdims=True)

                dL_H = cp.dot(dZo_H, dL_Zo) + cp.dot(dZi_H, dL_Zi) + cp.dot(dZc_H, dL_Zc)

                
            self.grad_Ui, self.grad_Wi, self.grad_bi, self.grad_Uo, self.grad_Wo, self.grad_bo, self.grad_Uc, self.grad_Wc, self.grad_bc = optimizer(
                dL_Ui, dL_Wi, dL_bi, dL_Uo, dL_Wo, dL_bo, dL_Uc, dL_Wc, dL_bc
            )
            
            self.Ui -= lr*(self.grad_Ui/cp.linalg.norm(self.grad_Ui))
            self.Wi -= lr*(self.grad_Wi/cp.linalg.norm(self.grad_Wi))
            self.bi -= lr*cp.clip(self.grad_bi, -5, 5)
            self.Uo -= lr*(self.grad_Uo/cp.linalg.norm(self.grad_Uo))
            self.Wo -= lr*(self.grad_Wo/cp.linalg.norm(self.grad_Wo))
            self.bo -= lr*cp.clip(self.grad_bo, -5, 5)
            self.Uc -= lr*(self.grad_Uc/cp.linalg.norm(self.grad_Uc))
            self.Wc -= lr*(self.grad_Wc/cp.linalg.norm(self.grad_Wc))
            self.bc -= lr*cp.clip(self.grad_bc, -5, 5)
            # self.Wi -= lr*self.grad_Wi
            # self.bi -= lr*self.grad_bi
            # self.Uo -= lr*self.grad_Uo
            # self.Wo -= lr*self.grad_Wo
            # self.bo -= lr*self.grad_bo
            # self.Uc -= lr*self.grad_Uc
            # self.Wc -= lr*self.grad_Wc
            # self.bc -= lr*self.grad_bc

    def getOutput(self):
        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def getName(self):
        pass

    def save(self):
        cp.save("weight_Ui.npy", self.Ui)
        cp.save("weight_Uo.npy", self.Uo)
        cp.save("weight_Uc.npy", self.Uc)
        cp.save("weight_Wi.npy", self.Wi)
        cp.save("weight_Wo.npy", self.Wo)
        cp.save("weight_Wc.npy", self.Wc)
        cp.save("bias_bi.npy", self.bi)
        cp.save("bias_bo.npy", self.bo)
        cp.save("bias_bc.npy", self.bc)

    def load(self):
        self.Ui = cp.load('weight_Ui.npy')
        self.Uo = cp.load('weight_Uo.npy')
        self.Uc = cp.load('weight_Uc.npy')
        self.Wi = cp.load('weight_Wi.npy')
        self.Wo = cp.load('weight_Wo.npy')
        self.Wc = cp.load('weight_Wc.npy')
        self.bi = cp.load('bias_bi.npy')
        self.bo = cp.load('bias_bo.npy')
        self.bc = cp.load('bias_bc.npy')

    def __call__(self, X):
        return self.forward(X)

#----------------------------------------------------------------------------------

class ModernLSTM(LSTM):
    def __init__(self, output_shape, input_shape=0, activation=None, recurrent_activation=None, initial_state=None, return_sequences=False, bias_initialize=True, **kwargs):
        super().__init__(output_shape, input_shape, activation)
        self.recurrent_activation = recurrent_activation
        self.return_sequences = return_sequences
        self.initial_state = initial_state
        self.bias_initialize = bias_initialize
        self.kwargs = kwargs
        self.Ui = None
        self.Wi = None
        self.Uo = None
        self.Wo = None
        self.Uc = None
        self.Wc = None
        self.bi = None
        self.bo = None
        self.bc = None
        self.Uf = None
        self.Wf = None
        self.bf = None

    def built(self):
        if self.build == False:
            self.Ui = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16)
            self.Wi = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16)
            self.Uo = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16)
            self.Wo = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16)
            self.Uc = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16)
            self.Wc = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16)
            self.Uf = cp.random.randn(self.input_shape[1], self.output_shape).astype(cp.float16)
            self.Wf = cp.random.randn(self.output_shape, self.output_shape).astype(cp.float16)
            if self.bias_initialize == True:
                self.bf = cp.random.uniform(1, 3, size=(1, self.output_shape))
                self.bi = cp.random.uniform(-3, -1, size=(1, self.output_shape))
                self.bo = cp.random.uniform(-3, -1, size=(1, self.output_shape))
                self.bc = cp.random.uniform(-0.2, 0.2, size=(1, self.output_shape))
            else: 
                self.bi = cp.random.randn(1, self.output_shape) * 0.2
                self.bo = cp.random.randn(1, self.output_shape) * 0.2
                self.bc = cp.random.randn(1, self.output_shape) * 0.2
                self.bf = cp.random.randn(1, self.output_shape) * 0.2
            self.grad_Ui = cp.zeros_like(self.Ui)
            self.grad_Wi = cp.zeros_like(self.Wi)
            self.grad_Uo = cp.zeros_like(self.Uo)
            self.grad_Wo = cp.zeros_like(self.Wo)
            self.grad_Uc = cp.zeros_like(self.Uc)
            self.grad_Wc = cp.zeros_like(self.Wc)
            self.grad_Uf = cp.zeros_like(self.Uf)
            self.grad_Wf = cp.zeros_like(self.Wf)
            self.grad_bi = cp.zeros_like(self.bi)
            self.grad_bo = cp.zeros_like(self.bo)
            self.grad_bc = cp.zeros_like(self.bc)
            self.grad_bf = cp.zeros_like(self.bf)
            self.build = True

    def forward(self, X):
        # I = input gate, O = output gate, C = cell state, H = hidden state
        # Zi = X.Ui + H.Wi + bi
        # I = recurrent_activation(Zi)
        # Zf = X.Uf + H.Wf + bf
        # F = recurrent_activation(Zf)
        # Zc = X.Uc + H.Wc + bc
        # Ctile = activation(Zc)
        # Zo = X.Uo + H.Wo + bo
        # O = recurrent_activation(Zo)
        # C = F*np + I*Ctile
        # H = O*activation(C)

        self.built()
        self.X = X
        self.I = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.O = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.F = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.Ctile = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.C = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        self.H = cp.zeros((X.shape[1]+1, X.shape[0], self.output_shape))
        if self.initial_state != None: self.H[0], self.C[0] = self.initial_state
        self.hidden_activation = [0]
        self.unit_activation = {'i': [0], 'o':[0], 'c':[0], 'f': [0]}

        for t in range(1,self.input_shape[0]+1):
            self.unit_activation['i'].append(copy.deepcopy(self.recurrent_activation))
            self.I[t] = self.unit_activation['i'][t](cp.dot(X[:,t-1,:], self.Ui) + cp.dot(self.H[t-1], self.Wi) + self.bi)
            self.unit_activation['f'].append(copy.deepcopy(self.recurrent_activation))
            self.F[t] = self.unit_activation['f'][t](cp.dot(X[:,t-1,:], self.Uf) + cp.dot(self.H[t-1], self.Wf) + self.bf)
            self.unit_activation['o'].append(copy.deepcopy(self.recurrent_activation))
            self.O[t] = self.unit_activation['o'][t](cp.dot(X[:,t-1,:], self.Uo) + cp.dot(self.H[t-1], self.Wo) + self.bo)
            self.unit_activation['c'].append(copy.deepcopy(self.activation))
            self.Ctile[t] = self.unit_activation['c'][t](cp.dot(X[:,t-1,:], self.Uc) + cp.dot(self.H[t-1], self.Wc) + self.bc)
            self.C[t] = self.F[t] * self.C[t-1] + self.I[t] * self.Ctile[t]
            self.hidden_activation.append(copy.deepcopy(self.activation))
            self.H[t] = self.O[t] *  self.hidden_activation[t](self.C[t])

        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def backward(self, dL_A, optimizer, lr):
        # df_C: derivative of activation function wrt C

        k = self.kwargs['truncated_step']

        if dL_A.ndim == 2:
            dL_H = dL_A
            dL_C = 0
            dL_X = 0
            dL_Ui = dL_Uo = dL_Uc = dL_Uf = 0
            dL_Wi = dL_Wo = dL_Wc = dL_Wf = 0
            dL_bi = dL_bo = dL_bc = dL_bf = 0

            for t in reversed(range(self.input_shape[0] + 1 - k, self.input_shape[0] + 1)):
                dZi_Ui = dZo_Uo = dZc_Uc = dZf_Uf = self.X[:,t-1,:]
                dZi_Wi = dZo_Wo = dZc_Wc = dZf_Wf = self.H[t-1]
                dZi_H = self.Wi
                dZo_H = self.Wo
                dZc_H = self.Wc
                dZf_H = self.Wf
                dC_np = self.F[t].T
                dC_F = self.C[t-1].T
                dC_I = self.Ctile[t].T
                dC_Ctile = self.I[t].T
                df_C = self.hidden_activation[t].deri
                dH_f = self.O[t].T
                dH_O = self.hidden_activation[t].a.T
                
                dL_C = dL_H * dH_f * df_C + dL_C*dC_np
                dL_Zi = (dL_C) * dC_I * self.unit_activation['i'][t].deri
                dL_Zo = (dL_H * dH_O) * self.unit_activation['o'][t].deri 
                dL_Zc = (dL_C) * dC_Ctile * self.unit_activation['c'][t].deri
                dL_Zf = (dL_C) * dC_F * self.unit_activation['f'][t].deri
                    
                dL_Ui += cp.dot(dL_Zi, dZi_Ui) 
                dL_Wi += cp.dot(dL_Zi, dZi_Wi) 
                dL_Uo += cp.dot(dL_Zo, dZo_Uo)    
                dL_Wo += cp.dot(dL_Zo, dZo_Wo)
                dL_Uc += cp.dot(dL_Zc, dZc_Uc)
                dL_Wc += cp.dot(dL_Zc, dZc_Wc)
                dL_Uf += cp.dot(dL_Zf, dZf_Uf)
                dL_Wf += cp.dot(dL_Zf, dZf_Wf)
                dL_bi += cp.sum(dL_Zi, axis=1, keepdims=True)
                dL_bo += cp.sum(dL_Zo, axis=1, keepdims=True)
                dL_bc += cp.sum(dL_Zc, axis=1, keepdims=True)
                dL_bf += cp.sum(dL_Zf, axis=1, keepdims=True)

                dL_H = cp.dot(dZo_H, dL_Zo) + cp.dot(dZi_H, dL_Zi) + cp.dot(dZc_H, dL_Zc) + cp.dot(dZf_H, dL_Zf)

                
            self.grad_Ui, self.grad_Wi, self.grad_bi, self.grad_Uo, self.grad_Wo, self.grad_bo, self.grad_Uc, self.grad_Wc, self.grad_bc, self.grad_Uf, self.grad_Wf, self.grad_bf = optimizer(
                dL_Ui, dL_Wi, dL_bi, dL_Uo, dL_Wo, dL_bo, dL_Uc, dL_Wc, dL_bc, dL_Uf, dL_Wf, dL_bf
            )
            
            self.Ui -= lr*(self.grad_Ui/cp.linalg.norm(self.grad_Ui))
            self.Wi -= lr*(self.grad_Wi/cp.linalg.norm(self.grad_Wi))
            self.bi -= lr*cp.clip(self.grad_bi, -5, 5)
            self.Uo -= lr*(self.grad_Uo/cp.linalg.norm(self.grad_Uo))
            self.Wo -= lr*(self.grad_Wo/cp.linalg.norm(self.grad_Wo))
            self.bo -= lr*cp.clip(self.grad_bo, -5, 5)
            self.Uc -= lr*(self.grad_Uc/cp.linalg.norm(self.grad_Uc))
            self.Wc -= lr*(self.grad_Wc/cp.linalg.norm(self.grad_Wc))
            self.bc -= lr*cp.clip(self.grad_bc, -5, 5)
            self.Uf -= lr*(self.grad_Uf/cp.linalg.norm(self.grad_Uf))
            self.Wf -= lr*(self.grad_Wf/cp.linalg.norm(self.grad_Wf))
            self.bf -= lr*cp.clip(self.grad_bf, -5, 5)

        return dL_X


    def getOutput(self):
        if self.return_sequences == True: return self.H
        else: return self.H[-1]

    def getName(self):
        pass

    def save(self):
        cp.save("weight_Ui.npy", self.Ui)
        cp.save("weight_Uo.npy", self.Uo)
        cp.save("weight_Uc.npy", self.Uc)
        cp.save("weight_Uf.npy", self.Uf)
        cp.save("weight_Wi.npy", self.Wi)
        cp.save("weight_Wo.npy", self.Wo)
        cp.save("weight_Wc.npy", self.Wc)
        cp.save("weight_Wf.npy", self.Wf)
        cp.save("bias_bi.npy", self.bi)
        cp.save("bias_bo.npy", self.bo)
        cp.save("bias_bc.npy", self.bc)
        cp.save("bias_bf.npy", self.bf)

    def load(self):
        self.Ui = cp.load('weight_Ui.npy')
        self.Uo = cp.load('weight_Uo.npy')
        self.Uc = cp.load('weight_Uc.npy')
        self.Uf = cp.load('weight_Uf.npy')
        self.Wi = cp.load('weight_Wi.npy')
        self.Wo = cp.load('weight_Wo.npy')
        self.Wc = cp.load('weight_Wc.npy')
        self.Wf = cp.load('weight_Wf.npy')
        self.bi = cp.load('bias_bi.npy')
        self.bo = cp.load('bias_bo.npy')
        self.bc = cp.load('bias_bc.npy')
        self.bf = cp.load('bias_bf.npy')

    def __call__(self, X):
        return self.forward(X)

#----------------------------------------------------------------------------------

class Embedding(Layer):
    def __init__(self, output_shape, input_shape=0, activation=None, embedded=False, vocab_size=None, max_lenght=None, trainable=False):
        super().__init__(output_shape, input_shape, activation)
        self.E = None
        self.embedded = embedded
        self.vocab_size = vocab_size
        self.max_lenght = max_lenght
        self.trainable = trainable

    def built(self):
        if self.build == False:
            self.E = cp.zeros(self.input_shape, self.output_shape)
            self.grad_E = cp.zeros_like(self.E)
            self.build = True

    def one_hot_encodeing(self, X):
        return cp.eye(self.vocab_size, dtype=cp.int8)[X]

    def forward(self, X):
        if self.embedded == True:
            self.built()
            self.X = self.one_hot_encodeing(X)
            self.A = self.activation(cp.dot(self.X, self.E))
        else:
            self.A = self.one_hot_encodeing(X)

        return self.A
    
    def backward(self, dL_A, optimizer, lr):
        if self.trainable:
            pass

    def getOutput(self):
        return self.A

    def __call__(self, X):
        return self.forward(X)

if __name__ == '__main__':
    # model = RNN(units=3, output_shape=2, input_shape=(3,2), activation=Tanh(), initial_state=True, return_sequences=False)

    X = cp.array([
        [[1, 2], [3, 4], [5, 6]],  # First sequence
        [[7, 8], [9, 10], [11, 12]],  # Second sequence
        [[13, 14], [15, 16], [17, 18]],  # Third sequence
        [[19, 20], [21, 22], [23, 24]]   # Fourth sequence
    ])
    Y = cp.random.randint(0, 1, (4, 3, 2))  # Binary class labels

    # print('U: ', model.U)
    # print('bU: ', model.bU)
    # print('W: ', model.W)
    # print('V: ', model.V)
    # print('bV: ', model.bV)
    # model.forward(X)
    # print(model.Y_hat[-1])
    print([i for i in reversed(range(1,4))])


    