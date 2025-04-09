# Scripts to train and evaluate models described in paper

```python main.py -m path_to_json_config ```

By default, models and logs will be saved under a new directory named results/ .

Evaluation is done via [eval.py](eval.py) (or for full resolution on the AbdomenCTCT dataset: [eval_full_res.py](eval_full_res.py)).

For more information:
-> ```python main.py -h ```
-> ```python eval.py -h ```
-> ```python eval_full_res.py -h ```

## Troubleshooting:
- _ _pickle.PicklingError: Can't pickle..._ ->  Change num_workers in main.py to 0 ([here](https://github.com/Kheil-Z/biomechanical_DLIR/edit/main/biomechanical_DLIR/main.py?plain=1#L210))
