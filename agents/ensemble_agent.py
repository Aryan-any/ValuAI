from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        
        # Initialize Specialist Agent (Handle potential Modal connection issues gracefuly)
        try:
            self.specialist = SpecialistAgent()
            self.has_specialist = True
        except Exception as e:
            self.log(f"Specialist Agent unavailable: {e}")
            self.specialist = None
            self.has_specialist = False
        
        # Frontier Agent (Core rquirement)
        self.frontier = FrontierAgent(collection)
        
        # Neural Network Agent (Handle missing model weights gracefuly)
        try:
            self.neural_network = NeuralNetworkAgent()
            self.has_neural_network = True
        except Exception as e:
            self.log(f"Neural Network Agent unavailable: {e}")
            self.neural_network = None
            self.has_neural_network = False
        
        self.preprocessor = Preprocessor()
        
        # Report which models are active
        active_models = ["Frontier"]
        if self.has_specialist:
            active_models.append("Specialist")
        if self.has_neural_network:
            active_models.append("NeuralNetwork")
        self.log(f"Ensemble Agent initialized with models: {', '.join(active_models)}")
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log("Pre-processing disabled (raw text used)")
        
        # Get predictions from available models
        frontier = self.frontier.price(rewrite)
        
        specialist = None
        if self.has_specialist:
            try:
                specialist = self.specialist.price(rewrite)
            except Exception as e:
                self.log(f"Specialist prediction failed: {e}")
        
        neural_network = None
        if self.has_neural_network:
            try:
                neural_network = self.neural_network.price(rewrite)
            except Exception as e:
                self.log(f"Neural Network prediction failed: {e}")
        
        # Dynamic ensemble weighting based on available models
        if self.has_specialist and self.has_neural_network and specialist and neural_network:
            # Full Ensemble: Frontier (80%), Specialist (10%), NN (10%)
            combined = frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
            self.log(f"Using 3-model ensemble: Frontier=${frontier:.2f}, Specialist=${specialist:.2f}, NN=${neural_network:.2f}")
        elif self.has_specialist and specialist:
            # Fallback A: Frontier (90%), Specialist (10%)
            combined = frontier * 0.9 + specialist * 0.1
            self.log(f"Using 2-model ensemble: Frontier=${frontier:.2f}, Specialist=${specialist:.2f}")
        elif self.has_neural_network and neural_network:
            # Fallback B: Frontier (90%), NN (10%)
            combined = frontier * 0.9 + neural_network * 0.1
            self.log(f"Using 2-model ensemble: Frontier=${frontier:.2f}, NN=${neural_network:.2f}")
        else:
            # Fallback C: Frontier Only (100%)
            combined = frontier
            self.log(f"Using Frontier Agent only: ${frontier:.2f}")
        
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined