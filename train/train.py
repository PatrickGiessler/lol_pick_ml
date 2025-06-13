from .fetcher import DataFetcher
from .trainer import ChampionTrainer

def main():
    fetcher = DataFetcher("http://localhost:3001/api/training/generate")
    x, y = fetcher.fetch_data()

    trainer = ChampionTrainer(input_dim=x.shape[1], output_dim=y.shape[1])
    trainer.train(x, y, epochs=10, batch_size=32)
    trainer.save("model/saved_model")

if __name__ == "__main__":
    main()