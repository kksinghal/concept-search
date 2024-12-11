# ConceptSearch

## Installation setup:
- Clone ARC-AGI benchmark in folder ./
    ```
    git clone git@github.com:fchollet/ARC-AGI.git
    ```
- SentenceTransformer for LLM-based scoring is too large to be uploaded in supplementary material, hence will be provided after review on Github. For now, the model can be fine-tuned using ./finetune_sentence_transformer.ipynb
- The entry point is ./main.py and it calls neural_search.py, where the ConceptSearch algorithm resides.
- To run the algorithm on the evaluation set:
    ```
    python main.py
    ```

## Notes
- Only a subset of logs are provided. The full set will be uploaded on Github after review.
