import transformers
from transformers import PretrainedConfig
import os
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch
import nltk
nltk.data.path.append("C:\\Users\\anush\\OneDrive\\Documents\\3.Ms_DSAI_20TO24\\UdS\Thesis\\Experiments\\Script\\eval_hal\\nltk_data")
nltk.download('punkt_tab', download_dir="C:\\Users\\anush\\OneDrive\\Documents\\3.Ms_DSAI_20TO24\\UdS\Thesis\\Experiments\\Script\\eval_hal\\nltk_data")
from nltk.tokenize import sent_tokenize #If you don't have nltk, you can use another sentence tokenizer (morphodita, spacy, etc.)
from tqdm import tqdm

class AlignScoreCS(transformers.XLMRobertaModel):
    """
    AlignScoreCS class
    Description:
        Model ALIGNSCORECS has been trained according the paper for 3 days on 4GPUs AMD NVIDIA.
        (3 epochs, 1e-5 learning rate, 1e-6 AdamWeps, batchsize 32, WarmupRatio 0.06, 0.1 WeighDecay)
        - XLMROBERTA-large model with 3 classification HEAD {regression,binary,3way} using shared encoder
        - trained on 7M docs incorporating various NLP tasks (QA,STS,Summarization,FactVer,InforRetrievel,NLI,Paraphrase..)
                - English and Czech translated datasets

    TRY:  .show_examples() to see some examples
    
    USAGE: AlignScore.py
        - .from_pretrained - loads the model, usage as transformers.model

        - .score(context, claim) - function
                - returns probs of the ALIGNED class using 3way class head as in the paper.
        
        - .classify(context, claim) - function
                - returns predicted class using bin class head as in the paper.
        
        alignScoreCS = AlignScoreCS.from_pretrained("/mnt/data/factcheck/AlignScore-data/AAmodel/MTLModel/mo
        alignScoreCS.score(context,claim)
        
        If you want to try different classification head use parameter:
            - task_name = "re" : regression head
            - task_name = "bin" : binary classification head
            - task_name = "3way" : 3way classification head
    """
    _regression_model = "re_model"
    _binary_class_model = "bin_model"
    _3way_class_model = "3way_model"
    
    def __init__(self, encoder, taskmodels_dict, model_name= "xlm-roberta-large", **kwargs):
        super().__init__(transformers.XLMRobertaConfig(), **kwargs)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
        self.model_name =  model_name
        self.tokenizer = None
        self.inferencer = None
        self.init_inferencer(device = "cpu")

    
    def init_inferencer(self, device = "cuda"):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name) if not self.tokenizer else self.tokenizer
        self.inferencer = self.InferenceHandler(self, self.tokenizer, device)
    
    """
        Score: scores the context and claim with Aligned probabitlity of given classification head
              - using altered code inferencer from ALignScore
            
            context   : list or str
            claim     : list or str
            eval_mode : {nli, bin, re, nli_sp, bin_sp or re_sp}
                    nli - 3way head
                    bin - 2way head
                    re - regression head
                (sp  - indicates whether to apply alignscore function chunk context and split claim into sentences
                        otherwise it truncates the text and returns probability of Aligned from that)
            eval_question_answer : list or str representing question if you want to evaluate context-answer question 
        DEFAULT: nli_sp
        Returns the consistency score (probability of Aligned class of 3-way head) between context text and claim text
         - using 2way classification head
    
    """
    def score(self, context, claim, eval_mode = "nli_sp", eval_question_answer = None, **kwargs):
            
        scores = self.inferencer.nlg_eval(context, claim, eval_mode=eval_mode, question = eval_question_answer)
        return scores

    
    """
        Classify: classify the context and claim to the class label given the eval model 
            context   : list or str
            claim     : list or str
            eval mode : {nli, bin, re, nli_sp, bin_sp or re_sp}
                    nli - 3way head
                    bin - 2way head
                    re - regression head
                (sp  - indicates whether to apply alignscore classification function chunk context and split claim into sentences
                       otherwise it truncates the text and returns predicted class)
        DEFAULT: bin_sp
        Returns the class of {Contradict, Aligned} between context text and claim text
         - using 2way classification head
    """
    def classify(self, context, claim, eval_mode = "bin_sp", **kwargs):
        eval_mode = eval_mode+"_cls" if ("cls" not in eval_mode) and ("class" not in eval_mode) else eval_mode
        scores = self.inferencer.nlg_eval(context, claim, eval_mode=eval_mode)
        return scores

    
    def forward(self, task_name = "3way", **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
    
    def __call__(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


    def to(self, device, **kwargs):
        self.init_inferencer(device = device)
        return super().to(device)

        
        return self
    
    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("XLMRoberta"):
            return "roberta"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")
    """
        pretrained_model_name_or_path    :str "krotima1/AlignScoreCS"       // but it is possible to use another NLI model but specify load_specific_head to 3way
                                            - path to the directory of AlignScoreCS
                                            - or pass "build_new" to create new multitask AlignScore architecture.
        load_specific_head               :str ["re", "bin", "3way"] or None // use this, and it will load only one architecture 
        load_another_model
    """  
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        model_name : str = "xlm-roberta-large",
        load_specific_head = None,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):

        architectures = [cls._3way_class_model, cls._regression_model, cls._binary_class_model] if load_specific_head is None else {"re": [cls._regression_model], "bin": [cls._binary_class_model], "3way" : [cls._3way_class_model]}[load_specific_head]
        is_3way_included = "3way" == load_specific_head or load_specific_head is None
        
        # DEPRECATED = it is here only because of loading previous versions... load from file
        if all(os.path.exists(os.path.join(pretrained_model_name_or_path, model_dir)) for model_dir in architectures):

            # Disables the warning
            transformers.logging.set_verbosity_error()

            shared_encoder = None
            taskmodels_dict = {}
            for path_name in tqdm(architectures,  desc='DEPRECATED: Loading architectures from a local directory'):
                task_name = path_name.split("_")[0]
                
                # Load the configuration for the task-specific model
                task_config = transformers.XLMRobertaConfig.from_json_file("{}/{}/config.json".format(pretrained_model_name_or_path,path_name))
                # Create the task-specific model
                model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, config=task_config,*model_args,**kwargs)
                # Load the weights for the task-specific model
                model.load_state_dict(torch.load("{}/{}/pytorch_model.bin".format(pretrained_model_name_or_path,path_name), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                # Set the shared encoder to the model's encoder
                if shared_encoder is None:
                    shared_encoder = getattr(model, AlignScoreCS.get_encoder_attr_name(model))
                else:
                    setattr(model, AlignScoreCS.get_encoder_attr_name(model), shared_encoder)
                taskmodels_dict[task_name] = model

            # Create the AlignScoreCS with the shared encoder and loaded task-specific models
            return AlignScoreCS(encoder=shared_encoder, taskmodels_dict=taskmodels_dict, model_name=model_name)
        # UP TO DATE LOADING FROM FILE:
        if all(os.path.exists(os.path.join(pretrained_model_name_or_path, {"3way_model" : "pytorch_model.bin"}.get(model_dir,model_dir))) for model_dir in architectures):
            shared_encoder = None
            taskmodels_dict = {}
            for path_name in tqdm(architectures,  desc='Loading architectures from a local directory'):
                task_name = path_name.split("_")[0]
                load_path = pretrained_model_name_or_path if task_name == "3way" else "{}/{}".format(pretrained_model_name_or_path,path_name)
                task_config = transformers.XLMRobertaConfig.from_json_file("{}/config.json".format(load_path))
                model = transformers.XLMRobertaForSequenceClassification.from_pretrained("{}".format(load_path), config=task_config,*model_args,**kwargs)
                if shared_encoder is None:
                    shared_encoder = getattr(model, AlignScoreCS.get_encoder_attr_name(model))
                else:
                    setattr(model, AlignScoreCS.get_encoder_attr_name(model), shared_encoder)
                taskmodels_dict[task_name] = model
            return AlignScoreCS(encoder=shared_encoder, taskmodels_dict=taskmodels_dict, model_name=model_name)
        # BUILD NEW AlignScoreCS
        if pretrained_model_name_or_path == "build_new":
            shared_encoder = None
            taskmodels_dict = {}
            for path_name in tqdm([cls._3way_class_model, cls._regression_model, cls._binary_class_model],  desc=f'Building new architectures from {model_name}'):
                task_name = path_name.split("_")[0]
                task_config = transformers.XLMRobertaConfig.from_pretrained(model_name)
                model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, config=task_config,*model_args,**kwargs)
                if shared_encoder is None:
                    shared_encoder = getattr(model, AlignScoreCS.get_encoder_attr_name(model))
                else:
                    setattr(model, AlignScoreCS.get_encoder_attr_name(model), shared_encoder)
                taskmodels_dict[task_name] = model
            return AlignScoreCS(encoder=shared_encoder, taskmodels_dict=taskmodels_dict, model_name=model_name)
            
        #LOADING FROM HUGGINGFACE HUB
        shared_encoder = None
        taskmodels_dict = {}
        for model_dir in tqdm(architectures,  desc='Loading from huggingface HUB'):
            task_name = model_dir.split("_")[0]
            subfolder = '' if task_name == "3way" else model_dir
            config = transformers.XLMRobertaConfig.from_pretrained(f"{pretrained_model_name_or_path}", subfolder=subfolder)
            model = transformers.XLMRobertaForSequenceClassification.from_pretrained(f"{pretrained_model_name_or_path}",config=config, subfolder=subfolder)
            if shared_encoder is None:
                shared_encoder = getattr(model, AlignScoreCS.get_encoder_attr_name(model))
            else:
                setattr(model, AlignScoreCS.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        alignScoreCS = AlignScoreCS(encoder=shared_encoder, taskmodels_dict=taskmodels_dict, model_name=model_name)
        return alignScoreCS
    
    """
        This saves the architectures into the directory. Model with 3way head is in the main dir, while bin and reg are in subfolders (bin_model, re_model).
    """
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        #It would be awesome to rewrite this to save only the classifier's head from taskmodels_dict and one encoder instead of 3x same encoder.
        #But who cares - only those who need save storage
        for task_name, model_type in self.taskmodels_dict.items():
            final_directory = Path(save_directory,task_name+"_model") if task_name in ["re", "bin"] else  Path(save_directory)
            model_type.save_pretrained(save_directory = final_directory,
                                    is_main_process = is_main_process,
                                    state_dict = state_dict,
                                    save_function = save_function,
                                    push_to_hub = push_to_hub,
                                    max_shard_size = max_shard_size,
                                    safe_serialization = safe_serialization,
                                    variant = variant,
                                    token = token,
                                    save_peft_format = save_peft_format,
                                    **kwargs)

    """
    This piece of code is copied and modified from AlignScore github code from: https://github.com/yuh-zha/AlignScore/blob/main/src/alignscore/inference.py
    ### eval_mode #### 
         ## sp ##     means chunk context to roughly 300 tokens and split claim into sentneces then apply AlignScore function to get consistency score
          - nli_sp      - ## DEFAULT ## returns consistency score of Aligned class from 3way head given context and claim using chunking
          - bin_sp      - returns consistency score of Aligned class from 2way head given context and claim using chunking
          - re_sp       - returns output of regression head given context and claim using chunking.
         ## simple ##  without chunking and splitting
          - nli         - returns probability of Aligned class from 3way head given context and claim.
          - bin         - returns probability of Aligned class from 2way head given context and claim.
          - re          - returns output of regression head given context and claim.
         ## sp_cls ##     means chunk context to roughly 300 tokens and split claim into sentneces then apply AlignScore Classification function to get class
          - nli_sp_cls      - returns class from 3way head given context and claim using chunking
          - bin_sp_cls      - returns class from 2way head given context and claim using chunking
          - re_sp_cls       - returns output from regression head given context and claim using chunking
         ## simple ##   without chunking and splitting
          - nli_cls         - returns class of Aligned class from 3way head given context and claim.
          - bin_cls         - returns class from 2way head given context and claim.
          - re_cls          - returns output of regression head given context and claim.
    """
    class InferenceHandler:
        def __init__(self, model, tokenizer, device = "cuda", batch_size = 32, verbose = False):
            # self.position_of_aligned_class = {"3way" : 1, "bin" : 1}
            self.input_evalmode_handler = {"3way_sp" : "nli_sp", "3way_sp_class" : "nli_sp_cls", "3way" : "nli", "3-way" : "nli", "3way_class" : "nli_cls",
                                 "2way_sp" : "bin_sp", "2way_sp_class" : "bin_sp_cls", "2way" : "bin", "2-way" : "bin", "2way_class" : "bin_cls",
                                 "reg_sp" : "re_sp", "reg_sp_class" : "re_sp_cls", "reg" : "re", "reg_class" : "re_cls"}
            self.taskname_handler = lambda eval_mode: "3way" if "nli" in eval_mode else ("bin" if "bin" in eval_mode else "re")
            #DEFAULT
            self.nlg_eval_mode = "nli_sp"
            self.task_name = "3way"

            #Model setup
            self.model = model
            self.device = device
            self.tokenizer = tokenizer
            # self.model.to(self.device)
            self.model.eval()
            
            self.batch_size = batch_size
            self.verbose = verbose
            
            self.softmax = nn.Softmax(dim=-1)

        
        def nlg_eval(self, premise, hypo, eval_mode = "nli_sp", question = None):
            
            if isinstance(premise, str) and isinstance(hypo, str):
                premise = [premise]
                hypo = [hypo]
                if (isinstance(question,str)):
                    question = [question]
            
            if question is None:
                question = [None]*len(premise)
                    
            #setup
            self.nlg_eval_mode = self.input_evalmode_handler.get(eval_mode, eval_mode)
            self.task_name = self.taskname_handler(self.nlg_eval_mode)
            assert self.nlg_eval_mode in set(self.input_evalmode_handler.values()), f"eval_mode is wrong {self.nlg_eval_mode}, use please : nli_sp or any other, look at the comments." 

            if "sp" in self.nlg_eval_mode:
                return self.inference_example_batch(premise, hypo, question)
            elif "sp" not in self.nlg_eval_mode:
                return self.inference(premise, hypo)
            return None
        
        def inference_example_batch(self, premise: list, hypo: list, question : list):
            """
            inference a example,
            premise: list
            hypo: list
            using self.inference to batch the process
            SummaC Style aggregation
            """
            self.disable_progress_bar_in_inference = True
            assert len(premise) == len(hypo), "Premise must has the same length with Hypothesis!"

            out_score = []
            for one_pre, one_hypo, one_quest in tqdm(zip(premise, hypo, question), desc="Evaluating", total=len(premise), disable=(not self.verbose)):
                out_score.append(self.inference_per_example(one_pre, one_hypo, one_quest))
            
            return torch.tensor(out_score)
        
        def inference_per_example(self, premise:str, hypo: str, quest = None):
            """
            inference a example,
            premise: string
            hypo: string
            using self.inference to batch the process
            """
            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield ' '.join(lst[i:i + n])
            
            premise_sents = sent_tokenize(premise)
            premise_sents = premise_sents or ['']

            n_chunk = len(premise.strip().split()) // 350 + 1
            n_chunk = max(len(premise_sents) // n_chunk, 1)
            premise_sents = [each for each in chunks(premise_sents, n_chunk)]

            hypo_sents = sent_tokenize(hypo)
            
            #add question to each sentence
            if quest is not None:
                hypo_sents = [quest+" "+ sent for sent in hypo_sents]

            premise_sent_mat = []
            hypo_sents_mat = []
            for i in range(len(premise_sents)):
                for j in range(len(hypo_sents)):
                    premise_sent_mat.append(premise_sents[i])
                    hypo_sents_mat.append(hypo_sents[j])  

            output_score = self.inference(premise_sent_mat, hypo_sents_mat) ### use NLI head OR ALIGN head
            if "cls" in self.nlg_eval_mode:
                output_score = output_score.view(len(premise_sents), len(hypo_sents),-1).mean(1).mean(0).argmax().item()
            else:
                output_score = output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item() ### sum or mean depends on the task/aspect
            return output_score

        def inference(self, premise, hypo):
            """
            inference a list of premise and hypo
            Standard aggregation
            """
            if isinstance(premise, str) and isinstance(hypo, str):
                premise = [premise]
                hypo = [hypo]
            
            batch = self.batch_tokenize(premise, hypo)
            output_score = []

            for mini_batch in tqdm(batch, desc="Evaluating", disable=not self.verbose or self.disable_progress_bar_in_inference):
                mini_batch = mini_batch.to(self.device)
                with torch.no_grad():
                    model_output = self.model.forward(task_name=self.task_name, **mini_batch)
                    model_output = model_output.logits
                    if self.task_name == "re":
                        model_output = model_output.cpu()
                        model_output = model_output[:,0]
                    else:
                        model_output = self.softmax(model_output).cpu()
                        if "cls" in self.nlg_eval_mode:
                            model_output = model_output
                            if "sp" not in self.nlg_eval_mode:
                                model_output = model_output.argmax(-1)
                        else:
                            model_output = model_output[:,1]
                            
                output_score.append(model_output)
            output_score = torch.cat(output_score)
            
            return output_score
    
        def batch_tokenize(self, premise, hypo):
            """
            input premise and hypos are lists
            """
            assert isinstance(premise, list) and isinstance(hypo, list)
            assert len(premise) == len(hypo), "premise and hypo should be in the same length."

            batch = []
            for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
                try:
                    mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
                except:
                    print('text_b too long...')
                    mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
                batch.append(mini_batch)

            return batch
        
        def chunks(self, lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
                
    def show_examples(self):
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        contexts = ["Jaromír Jágr (68) střelil poslední gól sezóny do branky Dominika Haška. Davy šílely dokonce i po celém zápase."]
        claims = ["Dav šílel, když Jarda (68) střelil gól.", "Dav šílel, když Jarda (78) střelil gól.", "Dav šílel jen při zápase, když Jarda (68) střelil gól.", "Dominik Hašek nedokázal chytit poslední střelu od Jágra.",
                 "Dominik Jágr (68) střelil poslední gól sezóny do branky Jaromíra Haška.", "Dominik Jágr (68) střelil poslední gól sezóny do branky Dominika Haška.", "Jaromír jágr nestřelil gól v sezóně.",
                  "Davy šílely, když střelily gól do branky Dominika Haška.","Davy šílely, když davy střelily gól do branky Dominika Haška.", "Dav šílel. Jarda střelil gól.", "Dav šílel. Jarda nestřelil gól.",
                 "Dneska odevzdávám diplomovou práci a koukám na hokej.", "Téma pojednává o hokeji", "Téma pojednává o baletu", "Dominik hašek je brankář", "Dominik hašek je útočník", "Jaromír Jágr je střelec", "Jaromír Jágr je hokejový útočník",
                  "Jaromír Jágr je hokejový brankář", "Na utkání se dívaly davy lidí, které byly potichu.", "Na utkání se dívaly davy lidí, které šílely."]
        print("EXAMPLES:")
        print("context:",contexts[0])
        print("SCORE: ", "claims:")
        for co, cl in zip(contexts*len(claims),claims):
            print(round(self.score(co,cl,eval_mode="nli_sp").tolist()[0],5),cl)
        print("EXAMPLES QA:")
        print("SCORE: ", "q-a pairs:")
        claims = [("Kdo střelil gól?", "Jaromír Jágr."), ("Kdo střelil gól?", "Domink Hašek."), ("Kdo nechytil střelu?", "Jaromír Jágr."), ("Kdo nechytil střelu?", "Domink Hašek.")
                  , ("Jaký má číslo drezu Jaromír Jágr?", "Jaromír Jágr má číslo drezu 68."), ("Kolik je Jaromíru Jágrovi let?", "Jaromíru Jágrovi je 68."), ("Kolik je Jaromíru Jágrovi let?", "Jaromíru Jágrovi je 67.")
                 , ("Co udělali lidi, když Jágr střelil gól?", "Lidi začali šílet. Dokonce šílely i po zápase."), ("Co udělali lidi, když Jágr střelil gól?", "Šli dát góla Haškovi")]
        for co,cl in zip(contexts*len(claims),claims):
            print(round(self.score(co, cl[1],eval_mode="nli_sp",eval_question_answer=cl[0] ).tolist()[0],5)," ".join(cl))