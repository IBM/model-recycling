# FAQ

## Why should I use a different base model than the vanilla pretrained model?

It improves results significantly in most cases, and in average. The best base RoBERTa-besa models improve results in 75% of the tasks we evaluated, with a median gain of 2.5 accuracy points. So if you had have to choose one base model, it would be best to use these top ranked models.

## Can I get worse results from training over the top ranked base model when compared to the vanilla model?

Yes. For example, in RoBERTa-base, about 1 in 4 tasks perform slightly better on the pretrained model. Furthermore, difference in seed randomization can yield variance in results. The best approach is to assess multiple models and evaluate on dev data. You can see example HuggingFace code that streamline this process.

## When shouldn't I use one of the recommended base models?

You should always review the base model license and fact sheet to ensure they meet your requirement for the particular use case.

## Which architectures are supported?

In the initial version, Roberta-base models are tracked. Other architectures will be added soon. Want us to add a specific model? Please [contact us](contact_us.md) and say so. If you have recommended training parameters, it is even better, send them too. 

## Could you test my model?

Sure. If the architecture is not supported, see the above question. You can add it to [HuggingFace](https://huggingface.co/docs/transformers/model_sharing#use-the-pushtohub-function)  and wait.
<!--    Really impatient? You can [contact us](contact_us.md) we don't make any promise.-->


## How frequently do you update the leaderboard?

We will update the results monthly.

## How do you assess the models?

We train a linear probing classification head for the MNLI on each candidate model.  We take each of the top 5 ranking models, and we fine-tune them on the 36 classification tasks (Consisting of sentiment, NLI, Twitter, topic classification and other general classification tasks).   We compare to the baseline of the vanilla model which is also trained and assessed on 5 seeds.   We use the following hyperparameters:

## I have another question.

Please [contact us](contact_us.md)
