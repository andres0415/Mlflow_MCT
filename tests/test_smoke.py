import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run(cmd):
    print('Running:', ' '.join(cmd))
    completed = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    print('stdout:', completed.stdout)
    print('stderr:', completed.stderr)
    assert completed.returncode == 0, f"Command failed: {' '.join(cmd)}"


def test_train_penguin_dry_run(tmp_path):
    run(['train_penguin_classifier.py', '--dry-run'])
    assert os.path.exists(os.path.join(ROOT, 'penguin_model_card.html'))


def test_register_generative_dry_run():
    run(['register_generative_model.py', '--dry-run'])
    assert os.path.exists(os.path.join(ROOT, 'generative_model_card.html'))


def test_create_legacy_and_api_cards():
    run(['create_legacy_model_card.py'])
    run(['create_api_model_card.py', '--name', 'test-model'])
    assert os.path.exists(os.path.join(ROOT, 'penguin_model_card_legacy.html'))
    assert os.path.exists(os.path.join(ROOT, 'generative_api_model_card.html'))
