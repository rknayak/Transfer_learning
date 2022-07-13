import os
import sys


def test_part1():
	os.chdir('part1_transferleanring')

	if not os.path.exists('galaxy_validation'):
		os.system('wget https://www.dropbox.com/s/4hn5gqeo28t155l/galaxy_validation.zip')
		os.system('unzip galaxy_validation.zip')
	path_to_ds = 'galaxy_validation/'

	from evaluate import evaluate_on_dataset

	accuracy = evaluate_on_dataset(path_to_ds)

	os.chdir('..')

	assert accuracy > 0.75


