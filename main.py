from build_dataset import build_dataset
from evaluation import run_evaluation

def main():
    dataset = build_dataset()
    df = run_evaluation(dataset)
    print("\nEvaluation Results:\n")
    print(df)

if __name__ == "__main__":
    main()