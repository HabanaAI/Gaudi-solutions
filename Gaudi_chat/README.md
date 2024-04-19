<small>Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.</small>

#### Licensed under the Apache License, Version 2.0 (the "License");
<small>You may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.</small>


# FastChat with Llama 2 on Intel&reg; Gaudi&reg; AI Accelerator
This example shows a fully interactive chatbot experience using the Llama 2 model published by Meta Labs, adapted to run on Gaudi2 and first-Gen Gaudi AI accelerators.  For the chat platform, we are using an open-source chatbot app and model server called FastChat published by LMSYS.   

The **meta-llama/Llama2-13B-chat-hf** model is a recent open-source LLM that is licensed for both research and commercial use. Llama 2 has 13 billion parameters which is much smaller compared 175 billion of GPT3, and even more for ChatGPT. This means faster and cheaper finetuning on your own data, and of course faster inference times as well.  Gaudi’s cost performance advantage over GPUs makes it even faster and more cost efficient. 

## Set Up the Environment 
Set up your cloud computing environment to get access to the first-gen Gaudi or Gaudi2 accelerator.  There are two options available in the cloud today:  

 - Intel Developer Cloud using Gaudi2 
   - https://developer.habana.ai/intel-developer-cloud 
   - You can access the Quick Start Video; a user account will need to be created.
     
 - Amazon EC2 DL1 Instances: based on first-gen Gaudi 
   - https://aws.amazon.com/ec2/instance-types/dl1/ 
   - Users can refer to Habana’s quick start guide [here](https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html) for instructions on how to start a DL1 instance; an AWS user account is required.  

Once you have an instance running, you will need to get the latest PyTorch Docker Image from the Habana Vault, see our [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more information. 

## Running the model 
To run the model, you will need to follow the instructions below. 

#### Start a Habana PyTorch Docker container 
```
docker pull vault.habana.ai/gaudi-docker/1.15.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host  vault.habana.ai/gaudi-docker/1.15.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
```
#### Clone the Gaudi-Solutions GitHub Repository to your $HOME directory and install the FastChat Application 
```
cd ~
git clone https://github.com/habanaai/Gaudi-Solutions 
cd Gaudi-Solutions/Gaudi_chat/ 
pip3 install --upgrade pip
pip3 install -e . 
```
#### How to access and Use the Llama 2 model
Use of the pretrained model is subject to compliance with third party licenses, including the “Llama 2 Community License Agreement” (LLAMAV2). For guidance on the intended use of the LLAMA2 model, what will be considered misuse and out-of-scope uses, who are the intended users and additional terms please review and read the instructions in this link https://ai.meta.com/llama/license/. Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.

To be able to run gated models like this Llama 2 model, you need the following:

* Have a HuggingFace account
* Agree to the terms of use of the model in its model card on the HF Hub
* set a read token
* Login to your account using the HF CLI: run huggingface-cli login before launching your script, please install the huggingface-hub `pip install -U "huggingface_hub[cli]"` if needed

`huggingface-cli login --token <your token here>`

Now, everything is set up and it’s time to chat!  We’ll launch the FastChat server application which will pull the Llama 2 model from the Hugging Face hub.  Note that the very first prompt will take a longer time becuase the Graph is being compiled for execution. 

Users acknowledge and understand that by downloading the model referenced herein they will be required to comply with third party licenses and rights pertaining to the model, and users will be solely liable and responsible for complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use or compliance with such third party licenses.

After you launch the FastChat Server you will see an “INST” instruction prompt, where you will be able to enter prompts and questions.   Type   `clear`  in the prompt to clear the queue and start over.   Hit the enter key on your keyboard at an empty prompt to stop the chat server.  

### Chat on Gaudi2 

In this case, we are using the meta-llama/Llama2-13B-chat-hf model from the HuggingFace Hub.  
```
python -m fastchat.serve.cli --model-path meta-llama/Llama-2-13b-chat-hf --device hpu --use_cache --temperature 0.7 --use_graphs --static_shapes --output_tps --conv-template llama2
```
```
config.json: 100% |█| 587/587 [00:00<00:00, 5.15MB/s]
tokenizer_config.json: 100%|█| 1.62k/1.62k [00:00<00:00, 18.3MB/s]
tokenizer.model: 100%|█| 500k/500k [00:00<00:00, 18.6MB/s]
special_tokens_map.json: 100%|█| 414/414 [00:00<00:00, 4.88MB/s]
tokenizer.json: 100%|█| 1.84M/1.84M [00:00<00:00, 9.85MB/s]
model.safetensors.index.json: 100%|█| 33.4k/33.4k [00:00<00:00, 9.15MB/s]
model-00001-of-00003.safetensors:  100%|█| 9.95G/9.95G [00:47<00:00, 211MB/s]
model-00002-of-00003.safetensors:  100%|█| 9.95G/9.95G [00:47<00:00, 211MB/s]
Downloading shards:   100%|█| 3/3 [02:10<00:00, 43.48s/it]
model-00003-of-00003.safetensors:  100%|█| 2.33G/9.95G [00:10<00:36, 207MB/s]

[INST]: Can you make an Itinerary for a trip to Orlando?  Make it 4 days long.
[/INST]: Of course! I'd be happy to help you plan your 4-day trip to Orlando. Here's an itinerary that includes a mix of popular theme parks, entertainment options, and some time for relaxation:
Day 1:
* Morning: Start your day at the Magic Kingdom, the most iconic of all the Disney parks. Experience classic attractions like Space Mountain: Mission 2, Splash Mountain, and the Haunted Mansion.
* Afternoon: Head over to Epcot, where you can explore the World Showcase, enjoy the Soarin' Around the World ride, and indulge in some delicious food and drinks from around the globe.
* Evening: End your day with the Happily Ever After fireworks display at Magic Kingdom.
Day 2:
* Morning: Begin your day at Universal Studios Florida, where you can experience thrilling rides like The Incredible Hulk Coaster and Hollywood Rip Ride Rockit. Don't miss out on the Wizarding World of Harry Potter – Diagon Alley, a must-see for any Harry Potter fan!
* Afternoon: Head over to Islands of Adventure, where you can explore the immersive world of Jurassic Park, take on the Incredible Hulk Coaster, and enjoy the thrilling rides and attractions of the Marvel Super Hero Island.
* Evening: End your day with a visit to Universal CityWalk, where you can dine, shop, and enjoy live entertainment.
Day 3:
* Morning: Spend your third day at SeaWorld Orlando, where you can see marine animals up close, including dolphins, sea lions, and penguins. Don't miss the Mako, Kraken, and Manta roller coasters!
* Afternoon: Head to Aquatica, SeaWorld's water park, where you can enjoy thrilling water slides, a wave pool, and a lazy river.
* Evening: End your day with a visit to Disney's Typhoon Lagoon or Blizzard Beach, two of Orlando's famous water parks.
Day 4:
* Morning: Start your final day with a visit to Disney's Hollywood Studios, where you can
-------------------------------------------
Time: 13.355    Tokens: 512     TPS: 38.34
-------------------------------------------
```
### Chat on First-gen Gaudi 
In this case, we are using the meta-llama/Llama2-7B-chat-hf model from the HuggingFace Hub.  You can use the 7B or 13B model.
```
python -m fastchat.serve.cli --model-path /root/data/llama_inference/Llama-2-7b-chat-hf --device hpu --use_cache --temperature 0.7 --use_graphs --static_shapes --output_tps --conv-template llama2
```
```
Downloading pytorch_model.bin:   100%|█| 5.68G/5.68G [4:00<04:00, 16.2MB/s] 
[INST]: Write a recipe to make chocolate chip cookies
[/INST]: Of course, I'd be happy to help you with that! Here's a classic recipe for chocolate chip cookies that yields delicious and chewy treats:

Ingredients:

* 1 cup (2 sticks) unsalted butter, at room temperature
* 3/4 cup granulated sugar
* 1 cup brown sugar
* 2 large eggs
* 2 teaspoons vanilla extract
* 2 1/4 cups all-purpose flour
* 1 teaspoon baking soda
* 1 teaspoon salt
* 2 cups semisweet chocolate chips

Instructions:

1. Preheat your oven to 375°F (190°C). Line a baking sheet with parchment paper or a silicone mat.
2. In a medium-sized bowl, whisk together the flour, baking soda, and salt. Set aside.
3. In a large bowl, use an electric mixer to cream together the butter and sugars until light and fluffy, about 2 minutes.
4. Beat in the eggs one at a time, followed by the vanilla extract.
5. Gradually mix in the flour mixture until just combined, being careful not to overmix.
6. Stir in the chocolate chips.
7. Drop rounded tablespoonfuls of the dough onto the prepared baking sheet, about 2 inches apart.
8. Bake for 10-12 minutes, or until the edges are lightly golden brown and the centers are set.
9. Remove the cookies from the oven and let them cool on the baking sheet for 5 minutes before transferring them to a wire rack to cool completely.

Enjoy your delicious homemade chocolate chip cookies!

I hope you find this recipe helpful and enjoy the process of baking these classic treats. If you have any questions or need further assistance, please don't hesitate to ask.
-------------------------------------------
Time: 19.112    Tokens: 469     TPS: 24.54
-------------------------------------------
```
## Next Steps 
Customers who want to use a chatbot like FastChat/Llama 2 with their own proprietary data and deploy LLMs on-premise can take advantage of Gaudi2’s performance, cost and power efficiency.  We’ve also demonstrated Gaudi2 inference performance on larger models such as the open source 176B parameter BLOOMz model; check out the blog [here](https://developer.habana.ai/blog/fast-inference-on-large-language-models-bloomz-on-habana-gaudi2-accelerator).   

You can sign up for access to Gaudi2 on the Intel Developer Cloud and experiment with FastChat/Llama2 
