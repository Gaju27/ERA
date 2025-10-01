 from pathlib import Path


 def main():
     base = Path('.')
     for d in [
         'data', 'checkpoints', 'logs', 'runs', 'results'
     ]:
         p = base / d
         p.mkdir(parents=True, exist_ok=True)
         (p / '.gitkeep').touch()
         print(f"Created {p}")


 if __name__ == '__main__':
     main()


