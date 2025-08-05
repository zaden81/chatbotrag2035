from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()


def load_gemma():
    model_id = "google/gemma-3-1b-it"
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

tokenizer, model = load_gemma()
device = "cuda" if torch.cuda.is_available() else "cpu"

title = "Leaning Tower of Pisa"
section = ""
content = "Prior to restoration work performed between 1990 and 2001, Leaning Tower of Pisa leaned at an angle of 5.5 degrees, but the tower now leans at about 3.99 degrees. This means the top of the tower is displaced horizontally 3.9 meters (12 ft 10 in) from the center."

input_text = (
    "Hãy tách đoạn văn sau thành các câu ngắn, rõ ràng, có ý nghĩa độc lập, "
    "Mỗi câu một dòng, không thêm nội dung mới, chỉ tách đoạn văn thành các câu ngắn có ý nghĩa, không thêm ghi chú, không bịa thêm:\n\n"
    f"{content}"
)


input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7, top_p=0.95, do_sample=False)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=== KẾT QUẢ TÁCH ĐOẠN ===")
for line in output_text.strip().split("\n"):
    if line.strip():
        print("-", line.strip())