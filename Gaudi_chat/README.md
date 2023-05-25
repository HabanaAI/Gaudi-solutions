<small>Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.</small>

#### Licensed under the Apache License, Version 2.0 (the "License");
<small>You may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.</small>


# FastChat with Dolly 2.0 on Habana Gaudi 
This example shows a fully interactive chatbot experience using the open-source Dolly 2.0 model published by Databricks, adapted to run on Gaudi2 and first-Gen Gaudi AI accelerators.  For the chat platform, we are using an open-source chatbot app and model server called FastChat published by LMSYS.   

The Dolly 2.0 model is a recent open-source LLM that is licensed for both research and commercial use. Dolly 2.0  has 12 billion parameters which is much smaller compared 175 billion of GPT3, and even more for ChatGPT. This means faster and cheaper finetuning on your own data, and of course faster inference times as well.  Gaudi’s cost performance advantage over GPUs makes it even faster and more cost efficient. 

## Set Up the Environment 
Set up your cloud computing environment to get access to the first-gen Gaudi or Gaudi2 accelerator.  There are two options available in the cloud today:  

 - Amazon EC2 DL1 Instances: based on first-gen Gaudi 
   - https://aws.amazon.com/ec2/instance-types/dl1/ 
   - Users can refer to Habana’s quick start guide [here](https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html) for instructions on how to start a DL1 instance; an AWS user account is required.  

 - Intel Developer Cloud using Gaudi2 
   - https://developer.habana.ai/intel-developer-cloud 
   - Instructions are provided on the Developer page; a user account will need to be created.   
   
Once you have an instance running, you will need to get the latest PyTorch Docker Image from the Habana Vault, see our [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more information. 

## Running the model 
To run the model, you will need to follow the instructions below. 

#### Start a Habana PyTorch Docker container 
```
docker pull vault.habana.ai/gaudi-docker/1.9.0/ubuntu20.04/habanalabs/pytorch-installer-1.13.1:latest
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host  vault.habana.ai/gaudi-docker/1.9.0/ubuntu20.04/habanalabs/pytorch-installer-1.13.1:latest
```
#### Clone the Gaudi-Solutions GitHub Repository to your $HOME directory and install the FastChat Application 
```
cd ~
git clone https://github.com/habanaai/Gaudi-Solutions 
cd Gaudi-Solutions/Gaudi_chat/ 
./install.sh  # this will install the FastChat application 
```
Now, everything is set up and it’s time to chat!  We’ll launch the FastChat server application which will pull the Dolly2.0 model from the Hugging Face hub.  Since Gaudi2 has 96GB of HBM memory, it can run the full 12B parameter model.   The First-gen Gaudi with 32GB, can run the 3B parameter model.  

After you launch the FastChat Server you will see an “### Instruction” prompt, where you will be able to enter prompts and questions.   Type   `clear`  in the prompt to clear the queue and start over.   Type  `enter` at an empty prompt to stop the chat server.  

### Chat on Gaudi2 

In this case, we are using the databricks/dolly-v2-12b model from the HuggingFace Hub.  
```
python -m fastchat.serve.cli --model-path databricks/dolly-v2-12b  --device hpu  --use_cache --temperature 0.7 --use_graphs --static_shapes  --output_tps --conv-template dolly
```

```
Downloading (…)lve/main/config.json: 100%█  819/819 [00:00<00:00, 62.0kB/s] 
Downloading (…)okenizer_config.json: 100%|█| 450/450 [00:00<00:00, 225kB/s] 
Downloading (…)/main/tokenizer.json: 100%|█| 2.11M/2.11M [00:00<00:00, 5.82MB/s] 
Downloading (…)cial_tokens_map.json: 100%|█| 228/228 [00:00<00:00, 136kB/s] 
Downloading pytorch_model.bin:   100%|█| 23.8G/23.8G [4:00<04:00, 16.2MB/s] 
### Instruction: Can you make an Itinerary for a trip to Orlando?  Make it 4 days long. 
### Response: Day 1: 
- Go to Universal Studios 
- Go to SeaWorld 
- Eat lunch at Shake Shack 
- Rattle your brain at The Harry Potter Experience 
- Watch the sunset over Orlando 

Day 2: 
- Go to Epcot 
- Visit the Future World, and ride Soarin' 
- Visit the World Showcase 
- Eat dinner at Paddlefish 

Day 3: 
- Go to Disney Springs 
- Shop at the Marketplace 
- Eat at Via Carota 

Day 4: 
- Go to Sea Life Aquarium 
- Eat dinner at Redz Barn 
- Watch the fireworks show at Universal Studios 
------------------------------------------- 
Time: 3.743     Tokens: 138     TPS: 36.87 
------------------------------------------- 
```
### Chat on First-gen Gaudi 
In this case, we are using the databricks/dolly-v2-3b model from the HuggingFace Hub. 
```
python -m fastchat.serve.cli --model-path databricks/dolly-v2-3b  --device hpu  --use_cache --temperature 0.7 --use_graphs --static_shapes  --output_tps  --conv-template dolly 
```

```
Downloading pytorch_model.bin:   100%|█| 5.68G/5.68G [4:00<04:00, 16.2MB/s] 
### Instruction: Write a recipe to make chocolate chip cookies 
### Response: 1/2 c softened butter 
4 tbs brown sugar 
2 eggs 
1 Tbs vanilla 
3/4 c white sugar 
2 c white chocolate chips 
2 c chocolate chips 
1 c semi sweet chips 
3 c all purpose flour 
1 t baking soda 
1/4 tsp salt 
1/2 tsp baking powder 
1/2 c semi sweet chocolate chips to garnish 
------------------------------------------- 
Time: 2.738     Tokens: 82      TPS: 29.95 
------------------------------------------- 
```
## Next Steps 
Customers who want to use a chatbot like FastChat/Dolly2.0 with their own proprietary data and deploy LLMs on-premise can take advantage of Gaudi2’s performance, cost and power efficiency.  We’ve also demonstrated Gaudi2 inference performance on larger models such as the open source 176B parameter BLOOMz model; check out the blog [here](https://developer.habana.ai/blog/fast-inference-on-large-language-models-bloomz-on-habana-gaudi2-accelerator).   

You can sign up for access to Gaudi2 on the Intel Developer Cloud and experiment with FastChat/Dolly2.0. 
