from typing import Optional, Union, Tuple, List, Sequence, Iterable

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models

from ..learning import PostPre
from ..network import Network
from ..network.nodes import Input, LIFNodes, DiehlAndCookNodes, AdaptiveLIFNodes
from ..network.topology import Connection, LocalConnection, SparseConnection


class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a
    fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization
            constant.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt

        self.add_layer(Input(n=self.n_inpt, traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNodes(
                n=self.n_neurons,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                w=w,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )


class MultimodalDiehlAndCook(Network):
  # language=rst
  """
  Implements the spiking neural network architecture from `(Diehl & Cook 2015)
  <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
  """

  def __init__(
        self,
        w_imag: torch.tensor, # le paso los pesos de Miguel por parámetro, la inicializacion se hace así ??
        w_audio: torch.tensor, # le paso los pesos de Isabel por parámetro
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
  
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``MultimodalDiehlAndCook``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        :param w_imag: pesos red imagen Miguel
        :param w_audio: pesos red audio Isabel
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.w_imag = w_imag
        self.w_audio = w_audio
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
       

######### LAYERS ###############

###### INPUT LAYER ##############

        """
        En esta capa tenemos tantas neuronas como número de píxeles totales tienen
        las imágenes que vamos a introducir en la red. Si unimos las imágenes de MNIST
        y las generadas por el audio, nos quedarían muestras nuevas de dimensiones 28x28 = 784 píxeles * 2 = 1568 px
        por tanto necesitaríamos un self.n_inpt = 1568 
        
        Para la variable self.inpt_shape para las imágenes originales de 28x28 teníamos:
        Shape of the input tensors (images of 28 x 28 pixels) inpt_shape=(1, 28, 28),
        así que supongo que ahora tendríamos inpt_shape = (1,56,28) ya que lo que estaríamos haciendo al final sería
        poner una imagen debajo de la otra, teniendo el doble de ancho el mismo largo
        
        """
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        
#################################
###### SAME MODEL FOR 2 EXCITATORY LAYERS ##############

        """
        En principo creo un modelo común a ambas aunque tengo que mirar los valores que usa Isabel para esta capa
        por si son distintos en la red de audio. Pero deberían ser los mismos param que usa Miguel
        
        """
        exc_layer_imag = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        
        exc_layer_audio = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        
        
#################################        
###### SAME MODEL FOR 2 INHIBITORY LAYERS ##############

        """
        En principo creo un modelo común a ambas aunque tengo que mirar los valores que usa Isabel para esta capa
        por si son distintos en la red de audio. Pero deberían ser los mismos param que usa Miguel
        
        """
        inh_layer_imag = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )
        
        inh_layer_audio = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )
        
################################# 
######### CONNECTIONS ###############

        """
        Tenemos dos opciones, o crear una capa exc única en la anulamos las conexiones de input->exc como nos conviene para tener dentro de una única capa exc dos entidades aisladas
        o creamos dos capas exc diferentes por lo que necesitaríamos tener 2 conexiones input_exc_conn distintas de dimensiones 1516xn_neurons. Para las posteriores conexiones multimodales
        mejor tener dos capas excitadoras diferentes porque no se si podrían unir nodos de una misma capa pero no lo veo posible por la arquitectura de Bindsnet. Lo único para recoger las 
        variables de estado (p.e. spikes) necesitaremos acceder a dos redes exc diferentes y crear sendos monitores para ellas.
        
        - Si quisiera una única capa excitadora:
        
        Tendremos inicialmente conexiones entre la capa INPUT y las dos capas excitadoras emulando los córtex visual y auditivo,
        esto es de los 1-758 px (de la neurona en pos = 0? a la pos = 757) las conexiones serán todas con todas con la capa excitadora nº1 (procesado imagen) y de los
        759-1516px (de la neurona en pos = 758? a la pos = 1515) las conexiones serán todas con todas con la capa excitadora nº2 (procesado audio).
        
        Como las conexiones son globales y la definición de su comportamiento se aplica a todos los nodos de la capa, definimos conexiones iniciales de todos
        los nodos de input con la capa1 y todos los nodos de input con la capa2. 
        
        A continuación de las neuronas de la 1 a la 758 pondremos a 0 todas las conexiones que vayan a la capa2 (esto es pesos[i<758][j>n_total_neurons_capa_exc_1-] = 0)
        y de la neurona 759 a la 1516 haremos lo propio con las conexiones que van a la capa1 (esto es pesos[i>758][j<n_total_neurons_capa_exc_1+1] = 0)
        
        - Si quiero tener dos capas excitadoras diferentes:
        
        Rellenaré la variable W primero con los pesos del córtex visual y crearé las conexiones, luego haré lo propio con las del córtex auditivo
        """
####### Connections ################
####### Conexiones globales iniciales entre INPUT -> EXC1 & EXC2 ##########

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)

        """
        Ahora tendremos que anular los pesos de las conexiones que no queremos como hemos indicado arriba y sustituir los valores obtenidos por Miguel e Isabel 
        Voy a suponer que tengo los pesos de miguel e isabel en dos arrays llamados w_imag y w_audio (tienen dimensiones de 758xn_neurons)
        
        """
############## CÓRTEX PRIMARIO ###################

        for i in range(self.n_inpt):  # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons): # Con esto me coge todas las neuronas de la capa exc1
            if i<758: # Aquí estoy procesando píxeles de MNIST
                w[i][j] = w_imag[i][j] # Asigno los pesos de Miguel para el número de neuronas dado
            else:
                w[i][j] = 0
                
                
        cortex_visual = Connection(
            source=input_layer,
            target=exc_layer_imag,
            w=w,
            # update_rule=PostPre, la comentamos porque ahora estos pesos son fijos
            #nu=nu,
            #reduction=reduction,
            wmin=wmin, # esto lo tengo que cambiar no?
            wmax=wmax, # esto lo tengo que cambiar no?
            #norm=norm,
        )     
        
        #Córtex visual creado
################################################

############## CÓRTEX AUDITIVO ###################

        for i in range(self.n_inpt): # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons*2): # Con esto me coge todas las conexiones con las neuronas de laexc2
             if i>757: # Aquí estoy procesando píxeles de audio convertidos a imagen
                 w[i][j] = w_audio[i-758][j] # Asigno los pesos de Isabel para el número de neuronas dado
             else:
                 w[i][j] = 0
        
        cortex_auditivo = Connection(
            source=input_layer,
            target=exc_layer_audio,
            w=w,
            # update_rule=PostPre, la comentamos porque ahora estos pesos son fijos
            #nu=nu,
            #reduction=reduction,
            wmin=wmin, # esto lo tengo que cambiar no?
            wmax=wmax, # esto lo tengo que cambiar no?
            #norm=norm,
        )     
        
        #Córtex auditivo creado
        
# Ya tengo aislados los dos córtex y asignados los valores de las conexiones válidas para los córtex visual y auditivo. Conexiones entre INPUT -> EXC1 & EXC2 terminadas 
################################################
####### Conexiones entre EXC1-> INH1 y EXC2-> INH2 mismo modelo para las dos redes ##########

        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        
        exc_inh_conn1 = Connection(
            source=exc_layer_imag, target=inh_layer_imag, w=w, wmin=0, wmax=self.exc
        )
        exc_inh_conn2 = Connection(
            source=exc_layer_audio, target=inh_layer_audio, w=w, wmin=0, wmax=self.exc
        )
        
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn1 = Connection(
            source=inh_layer_imag, target=exc_layer_imag, w=w, wmin=-self.inh, wmax=0
        )
        
        inh_exc_conn2 = Connection(
            source=inh_layer_audio, target=exc_layer_audio, w=w, wmin=-self.inh, wmax=0
        )
########################################################
############# CONEXIONES MULTIMODALES ##################
        """
        Primero conecto todos los nodos de la capa excitadora 1 con los de la excitadora 2
        Y los inicializo todos a 0. Las conexiones son de todos con todos en ambos sentidos.
        Se quitará el valor de 0 cuando veamos los assignments y unamos nodos, y nunca se podrá
        tener una conexión bidireccional entre ambos nodos.
        """
        w = torch.zeros(self.n_neurons, self.n_neurons) # inicializados a 0

        exc1_exc2_conn = Connection(
            source=exc_layer_imag,
            target=exc_layer_audio,
            w=w,
            update_rule=PostPre, # Pesos entrenables
            nu=nu,
            reduction=reduction,
            wmin=wmin, # esto lo tengo que cambiar no?
            wmax=wmax, # esto lo tengo que cambiar no?
            norm=norm,
        )   

        exc2_exc1_conn =  Connection(
            source=exc_layer_audio,
            target=exc_layer_imag,
            w=w,
            update_rule=PostPre, # Pesos entrenables
            nu=nu,
            reduction=reduction,
            wmin=wmin, # esto lo tengo que cambiar no?
            wmax=wmax, # esto lo tengo que cambiar no?
            norm=norm,
        )  

########################################################


        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer_imag, name="Ae_imag")
        self.add_layer(inh_layer_imag, name="Ai_imag")
        self.add_layer(exc_layer_audio, name="Ae_audio")
        self.add_layer(inh_layer_audio, name="Ai_audio")
        
        self.add_connection(cortex_visual, source="X", target="Ae_imag")
        self.add_connection(cortex_auditivo, source = "X", target = "Ae_audio")
        
        self.add_connection(exc_inh_conn1, source="Ae_imag", target="Ai_imag")
        self.add_connection(inh_exc_conn1, source="Ai_imag", target="Ae_imag")
        
        self.add_connection(exc_inh_conn2, source="Ae_audio", target="Ai_audio")
        self.add_connection(inh_exc_conn2, source="Ai_audio", target="Ae_audio")
        
        self.add_connection(exc1_exc2_conn, source = "Ae_imag", target="Ae_audio")
        self.add_connection(exc2_exc1_conn, source = "Ae_audio", target="Ae_imag")


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
    
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
        


class DiehlAndCook2015v2(Network):
    # language=rst
  """
  Implements the spiking neural network architecture from `(Diehl & Cook 2015)
  <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
  """

  def __init__(
        self,
        w_imag: torch.tensor, # valor posible de inicialización de los pesos para conexiones entrada-córtex imagen
        w_audio: torch.tensor, # valor posible de inicialización de los pesos para conexiones entrada-córtex auditivo
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
  
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``MultimodalDiehlAndCook``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        :param w_imag: pesos red imagen Miguel
        :param w_audio: pesos red audio Isabel
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.w_imag = w_imag
        self.w_audio = w_audio
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
       

######### LAYERS ###############

###### INPUT LAYER ##############

        """
        En esta capa tenemos tantas neuronas como número de píxeles totales tienen
        las imágenes que vamos a introducir en la red. Si unimos las imágenes de MNIST
        y las generadas por el audio, nos quedarían muestras nuevas de dimensiones 28x28 = 784 píxeles * 2 = 1568 px
        por tanto necesitaríamos un self.n_inpt = 1568 
        
        Para la variable self.inpt_shape para las imágenes originales de 28x28 teníamos:
        Shape of the input tensors (images of 28 x 28 pixels) inpt_shape=(1, 28, 28),
        así que ahora tendríamos inpt_shape = (1,56,28) ya que lo que estaríamos haciendo al final sería
        poner una imagen debajo de la otra, teniendo el doble de ancho el mismo largo
        
        """
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        
#################################
###### SAME MODEL FOR 2 EXCITATORY LAYERS ##############

        """
        En principo creo un modelo común a ambas aunque tengo que mirar los valores que usa Isabel para esta capa
        por si son distintos en la red de audio. Pero deberían ser los mismos param que usa Miguel
        
        """
        exc_layer_imag = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        
        exc_layer_audio = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        
        
#################################        
###### SAME MODEL FOR 2 INHIBITORY LAYERS ##############

        """
        En principo creo un modelo común a ambas aunque tengo que mirar los valores que usa Isabel para esta capa
        por si son distintos en la red de audio. Pero deberían ser los mismos param que usa Miguel
        
        """
        inh_layer_imag = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )
        
        inh_layer_audio = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )
        
################################# 
######### CONNECTIONS ###############

        """
        Tenemos dos opciones, o crear una capa exc única en la anulamos las conexiones de input->exc como nos conviene para tener dentro de una única capa exc dos entidades aisladas
        o creamos dos capas exc diferentes por lo que necesitaríamos tener 2 conexiones input_exc_conn distintas de dimensiones 1516xn_neurons. Para las posteriores conexiones multimodales
        mejor tener dos capas excitadoras diferentes porque no se si podrían unir nodos de una misma capa pero no lo veo posible por la arquitectura de Bindsnet. Lo único para recoger las 
        variables de estado (p.e. spikes) necesitaremos acceder a dos redes exc diferentes y crear sendos monitores para ellas.
        
        - Si quisiera una única capa excitadora:
        
        Tendremos inicialmente conexiones entre la capa INPUT y las dos capas excitadoras emulando los córtex visual y auditivo,
        esto es de los 1-758 px (de la neurona en pos = 0? a la pos = 757) las conexiones serán todas con todas con la capa excitadora nº1 (procesado imagen) y de los
        759-1516px (de la neurona en pos = 758? a la pos = 1515) las conexiones serán todas con todas con la capa excitadora nº2 (procesado audio).
        
        Como las conexiones son globales y la definición de su comportamiento se aplica a todos los nodos de la capa, definimos conexiones iniciales de todos
        los nodos de input con la capa1 y todos los nodos de input con la capa2. 
        
        A continuación de las neuronas de la 1 a la 758 pondremos a 0 todas las conexiones que vayan a la capa2 (esto es pesos[i<758][j>n_total_neurons_capa_exc_1-] = 0)
        y de la neurona 759 a la 1516 haremos lo propio con las conexiones que van a la capa1 (esto es pesos[i>758][j<n_total_neurons_capa_exc_1+1] = 0)
        
        - Si quiero tener dos capas excitadoras diferentes:
        
        Rellenaré la variable W primero con los pesos del córtex visual y crearé las conexiones, luego haré lo propio con las del córtex auditivo
        """
####### Connections ################
####### Conexiones globales iniciales entre INPUT -> EXC1 & EXC2 ##########

        

        """
        Ahora tendremos que anular los pesos de las conexiones que no queremos como hemos indicado arriba y sustituir los valores obtenidos por Miguel e Isabel 
        Voy a suponer que tengo los pesos de miguel e isabel en dos arrays llamados w_imag y w_audio (tienen dimensiones de 758xn_neurons)
        
        """
############## CÓRTEX VISUAL ###################
        """
        for i in range(self.n_inpt):  # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons): # Con esto me coge todas las neuronas de la capa exc1
            if i<784: # Aquí estoy procesando píxeles de MNIST
                w[i][j] = w_imag[i][j] # Asigno los pesos fijos
            else:
                w[i][j] = 0
             
        """

        
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)

        for i in range(self.n_inpt):  # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons): # Con esto me coge todas las neuronas de la capa exc1
            if i>783: # Aquí estoy procesando píxeles de MNIST
                w[i][j] = 0 # Asigno los pesos 0 para aislar que la info de audio llegue al córtex visual
        

        cortex_visual = Connection(
            source=input_layer,
            target=exc_layer_imag,
            w=w,
            update_rule=PostPre, #la comentamos si los pesos son fijos/no queremos entrenamiento de este tipo de conexiones
            nu=nu,
            reduction=reduction,
            wmin=wmin, 
            wmax=wmax, 
            norm=norm,
        )     
        
        #Córtex visual creado
################################################

############## CÓRTEX AUDITIVO ###################
        """"
        for i in range(self.n_inpt): # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons): # Con esto me coge todas las conexiones con las neuronas de laexc2
             if i>783: # Aquí estoy procesando píxeles de audio convertidos a imagen
                 w[i][j] = w_audio[i-784][j] # Asigno los pesos de Isabel para el número de neuronas dado
             else:
                 w[i][j] = 0
      

        """
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons) # 1568 * 100 = 156.800 conexiones  

        for i in range(self.n_inpt):  # Con esto me coge desde la i=0 hasta la i=1515, todos los píxeles de la capa INPUT
          for j in range(self.n_neurons): # Con esto me coge todas las neuronas de la capa exc1
            if i<784: # Aquí estoy procesando píxeles de MNIST
                w[i][j] = 0 # Asigno los pesos 0 para aislar que la info de imagen llegue al córtex auditivo
    
        cortex_auditivo = Connection(
            source=input_layer,
            target=exc_layer_audio,
            w=w,
            update_rule=PostPre, #la comentamos si los pesos son fijos/no queremos entrenamiento de este tipo de conexiones
            nu=nu,
            reduction=reduction,
            wmin=wmin, 
            wmax=wmax, 
            norm=norm,
        )     
        
        #Córtex auditivo creado
        
# Ya tengo aislados los dos córtex y asignados los valores de las conexiones válidas para los córtex visual y auditivo. Conexiones entre INPUT -> EXC1 & EXC2 terminadas 
################################################
####### Conexiones entre EXC1-> INH1 y EXC2-> INH2 mismo modelo para las dos redes ##########

        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        
        exc_inh_conn1 = Connection(
            source=exc_layer_imag, target=inh_layer_imag, w=w, wmin=0, wmax=self.exc
        )
        exc_inh_conn2 = Connection(
            source=exc_layer_audio, target=inh_layer_audio, w=w, wmin=0, wmax=self.exc
        )
        
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn1 = Connection(
            source=inh_layer_imag, target=exc_layer_imag, w=w, wmin=-self.inh, wmax=0
        )
        
        inh_exc_conn2 = Connection(
            source=inh_layer_audio, target=exc_layer_audio, w=w, wmin=-self.inh, wmax=0
        )
        
########################################################
############# CONEXIONES MULTIMODALES ##################
        """
        Primero conecto todos los nodos de la capa excitadora 1 con los de la excitadora 2
        Y los inicializo todos a 0. Las conexiones son de todos con todos en ambos sentidos.
        Se quitará el valor de 0 cuando veamos los assignments y unamos nodos, y nunca se podrá
        tener una conexión bidireccional entre ambos nodos.

        

        #w = torch.zeros(self.n_neurons, self.n_neurons) # inicializados a 

        w = 0.3 * torch.rand(self.n_neurons, self.n_neurons) # N*N conexiones

        exc1_exc2_conn = Connection(
            source=exc_layer_imag,
            target=exc_layer_audio,
            w=w,
            update_rule=PostPre, #la comentamos si los pesos son fijos/no queremos entrenamiento de este tipo de conexiones
            nu=nu,
            reduction=reduction,
            wmin=wmin, 
            wmax=wmax, 
            norm=norm,
        )  

        exc2_exc1_conn = Connection(
            source=exc_layer_audio,
            target=exc_layer_imag,
            w=w,
            update_rule=PostPre, #la comentamos si los pesos son fijos/no queremos entrenamiento de este tipo de conexiones
            nu=nu,
            reduction=reduction,
            wmin=wmin, 
            wmax=wmax, 
            norm=norm,
        )  
      
      """
        
########################################################


        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer_imag, name="Ae_imag")
        self.add_layer(inh_layer_imag, name="Ai_imag")
        self.add_layer(exc_layer_audio, name="Ae_audio")
        self.add_layer(inh_layer_audio, name="Ai_audio")
        
        self.add_connection(cortex_visual, source="X", target="Ae_imag")
        self.add_connection(cortex_auditivo, source = "X", target = "Ae_audio")
        
        self.add_connection(exc_inh_conn1, source="Ae_imag", target="Ai_imag")
        self.add_connection(inh_exc_conn1, source="Ai_imag", target="Ae_imag")
        
        self.add_connection(exc_inh_conn2, source="Ae_audio", target="Ai_audio")
        self.add_connection(inh_exc_conn2, source="Ai_audio", target="Ae_audio")

        """
        self.add_connection(exc1_exc2_conn, source = "Ae_imag", target="Ae_audio")
        self.add_connection(exc2_exc1_conn, source = "Ae_audio", target="Ae_imag")
        """

class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture
    from `(Hazan et al. 2018) <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt
        self.inpt_shape = inpt_shape

        input_layer = Input(
            n=self.n_input, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_output_conn, source="X", target="Y")

        # add internal inhibitory connections
        w = torch.ones(self.n_neurons, self.n_neurons) - torch.diag(
            torch.ones(self.n_neurons)
        )
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    w[i, j] = np.sqrt(euclidean([x1, y1], [x2, y2]))
        w = w / w.max()
        w = (w * self.max_inhib) + self.start_inhib
        recurrent_output_conn = Connection(
            source=self.layers["Y"], target=self.layers["Y"], w=w
        )
        self.add_connection(recurrent_output_conn, source="Y", target="Y")


class LocallyConnectedNetwork(Network):
    # language=rst
    """
    Defines a two-layer network in which the input layer is "locally connected" to the
    output layer, and the output layer is recurrently inhibited connected such that
    neurons with the same input receptive field inhibit each other.
    """

    def __init__(
        self,
        n_inpt: int,
        input_shape: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        inh: float = 25.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: Optional[float] = 0.2,
    ) -> None:
        # language=rst
        """
        Constructor for class ``LocallyConnectedNetwork``. Uses ``DiehlAndCookNodes`` to
        avoid multiple spikes per timestep in the output layer population.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param input_shape: Two-dimensional shape of input population.
        :param kernel_size: Size of input windows. Integer or two-tuple of integers.
        :param stride: Length of horizontal, vertical stride across input space. Integer
            or two-tuple of integers.
        :param n_filters: Number of locally connected filters per input region. Integer
            or two-tuple of integers.
        :param inh: Strength of synapse weights from output layer back onto itself.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param norm: ``Input`` to ``DiehlAndCookNodes`` layer connection weights
            normalization constant.
        """
        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_inpt = n_inpt
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        input_layer = Input(n=self.n_inpt, traces=True, tc_trace=20.0)

        output_layer = DiehlAndCookNodes(
            n=self.n_filters * conv_size[0] * conv_size[1],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        input_output_conn = LocalConnection(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")
