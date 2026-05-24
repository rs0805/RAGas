# %%
from build_dataset import build_dataset
from evaluation import run_evaluation
from IPython.display import display

# %%
dataset = build_dataset()
df = run_evaluation(dataset)

# %%
styled = (
    df.style
    .background_gradient(
        subset=[
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy"
        ],
        cmap="RdYlGn"
    )
)

display(styled)
# %%
