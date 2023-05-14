import torch 
import torch.nn as nn 

from dataset import get_data_loader

# this function returns probabilities for every test case.
def test(model, test_df, tokenizer,  device):
    predictions = torch.empty(0).to(device, dtype=torch.float)
    

    test_data_loader = get_data_loader(
        df = test_df, 
        batch_size= 512, 
        tokenizer=tokenizer
    )
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_data_loader):
            ids = batch["ids"]
            mask = batch["mask"]
            token_type_ids = batch["token_type_ids"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            outputs = model(ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            predictions = torch.cat((predictions, nn.Sigmoid()(outputs)))
    
    return predictions.cpu().numpy().squeeze()




def eval(model, tokenizer, first_question, second_question, device):
    
    inputs = tokenizer.encode_plus(
        first_question,
        second_question,
        add_special_tokens=True,
    )

    ids = torch.tensor([inputs["input_ids"]], dtype=torch.long).to(device, dtype=torch.long)
    mask = torch.tensor([inputs["attention_mask"]], dtype=torch.long).to(device, dtype=torch.long)
    token_type_ids = torch.tensor([inputs["token_type_ids"]], dtype=torch.long).to(device, dtype=torch.long)

    with torch.no_grad():
        model.eval()
        output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        prob = nn.Sigmoid()(output).item()

        print("questions [{}] and [{}] are {} with score {}".format(first_question, second_question, 'similar' if prob > 0.5 else 'not similar', prob))

        