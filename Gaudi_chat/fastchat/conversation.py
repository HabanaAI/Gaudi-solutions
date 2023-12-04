import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA2 = auto()
    DOLLY = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system:
                ret = self.system
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


conv_one_shot = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_koala_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_dolly = Conversation(
    system=
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=('### Instruction', '### Response'),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

conv_dolly2 = Conversation(
    system=
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=('### Instruction', '### Response'),
    messages=(
        ("### Instruction", "Can you make an itinerary for a 4 day trip to San Diego?"),
        ("### Response", """
Day 1:

* Arrive in San Diego and check into your hotel
* Visit the San Diego Zoo, which is home to a wide variety of animals from around the world, including pandas, koalas, and giraffes.
* In the evening, head to the Gaslamp Quarter, a vibrant neighborhood filled with restaurants, bars, and live music venues.

Day 2:

* Start the day with a visit to Balboa Park, a large urban park that is home to a number of museums, gardens, and other attractions, including the San Diego Zoo and the San Diego Air & Space Museum.
* After lunch, head to La Jolla, a picturesque coastal neighborhood that is home to a variety of shops, restaurants, and beaches. You can take a stroll along the beach or visit the La Jolla Cove, a popular spot for swimming and snorkeling.
* In the evening, catch a show at the San Diego Repertory Theatre or the La Jolla Playhouse.

Day 3:

* Spend the day at the beach! San Diego is known for its beautiful beaches, and there are many to choose from, including Mission Beach, Pacific Beach, and Ocean Beach. You can spend the day soaking up the sun, swimming, surfing, or trying your hand at other water sports.
* In the evening, head to the Little Italy neighborhood, which is known for its trendy restaurants, bars, and shops. You can grab dinner at one of the many Italian restaurants or enjoy a glass of wine at a local tasting room.

Day 4:

* Start the day with a visit to the USS Midway Museum, which is housed on a former aircraft carrier and features a number of exhibits and interactive displays that tell the story of the ship and its crew.
* After lunch, head to Coronado Island, which is home to a number of attractions, including the Hotel del Coronado, a historic hotel that has hosted celebrities and presidents, and the Coronado Island Maritime Museum, which tells the story of the island's naval history.
* In the evening,""")
    ),
    offset=2,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

conv_dolly3 = Conversation(
    system=
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=('### Instruction', '### Response'),
    messages=(
        ("### Instruction", "Can you make an itinerary for a 4 day trip to San Diego?"),
        ("### Response", """Sure, here are some ideas for a trip to San Diego:
Day 1: Go to the coast and visit the San Diego naval base. The base is close to a number of beaches with sunbathing areas, and you can also visit the beach at San Onofre if you want a day at the beach away from the city.
Day 2: Spend the day in the city visiting the San Diego Air and Space Museum, Maritime Museum, and Balboa Park.
Day 3: Take a day trip to Los Angeles, about a 2 hour drive away. You can visit the city's museums and parks, and you can also see the Hollywood Sign and other landmarks.
Day 4: Take a day trip to Santa Barbara, about a 2 hour drive away. You can visit the city's beaches and museums, and you can also see the San Marcos Bridge and University of California campus.
You could also spend the night in San Francisco instead of Los Angeles or Santa Barbara, and then drive back to San Diego the next day. Overall, this would be a fun trip with plenty of options to spend time in the cities or on the beach, and you could even visit multiple national landmarks.
Please let me know if you have any other questions.""")
    ),
    offset=2,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
conv_llama2 = Conversation(
    system="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n",
    roles=("[INST]", "[/INST]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA2,
    sep=" ",
    sep2=" </s><s>",
    # sep2=" ",
)

conv_templates = {
    "conv_one_shot": conv_one_shot,
    "vicuna_v1.1": conv_vicuna_v1_1,
    "koala_v1": conv_koala_v1,
    "dolly": conv_dolly,
    "dolly2": conv_dolly2,
    "dolly3": conv_dolly3,
    "llama2": conv_llama2
}


def get_default_conv_template(model_name):
    model_name = model_name.lower()
    if "vicuna" in model_name or "output" in model_name:
        return conv_vicuna_v1_1
    elif "koala" in model_name:
        return conv_koala_v1
    elif "dolly" in model_name:
        return conv_dolly
    return conv_one_shot


if __name__ == "__main__":
    print(default_conversation.get_prompt())
