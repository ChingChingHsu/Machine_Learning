"""
File: validEmailAddress_2.py
Name: ching-ching
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  There is only one  ’@’ in the string
feature2:  There are no whitespace in the string
feature3:  There are no consecutive dots in email address
feature4:  Some strings before '@’
feature5:  Some strings after '@’
feature6:  The string after '@’ contains a '.'
feature7:  The string before '@’does not start or end with a '.'
feature8:  Ends with '.com'
feature9:  Ends with '.edu'
feature10: Ends with '.tw'

Accuracy of your model: 0.8846153846153846
"""

import numpy as np


WEIGHT = [                           # The weight vector selected by you
	[3.0],                              # (Please fill in your own weights)
	[3.0],
	[4.0],
	[1.5],
	[1.5],
	[2.0],
	[1.5],
	[0.1],
	[0.1],
	[0.1]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	correct_count = 0
	weight_vector = np.array(WEIGHT)
	for i, maybe_email in enumerate(maybe_email_list):
		feature_vector = feature_extractor(maybe_email)
		weight_vector_transposed = weight_vector.T
		score_matrix = weight_vector_transposed.dot(feature_vector)
		score = score_matrix[0][0]
		prediction = 1 if score > 16.5 else 0
		if prediction == 1:
			print(f'"{maybe_email}" -> Prediction: Valid')
		else:
			print(f'"{maybe_email}" -> Prediction: Invalid')

		is_actually_valid = 1 if i >= 13 else 0
		if prediction == is_actually_valid:
			correct_count += 1
	accuracy = correct_count / len(maybe_email_list)
	print(f'Accuracy: {accuracy}')



def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = np.zeros((len(WEIGHT), 1))

	has_single_at = maybe_email.count('@') == 1
	feature_vector[0][0] = 1 if has_single_at else 0  # single'@'
	feature_vector[1][0] = 1 if ' ' not in maybe_email else 0
	feature_vector[2][0] = 1 if '..'not in maybe_email and '()'not in maybe_email and '"'not in maybe_email else 0

	if maybe_email.count('@') == 1:
		local_part, domain_part = maybe_email.split('@')
		feature_vector[3][0] = 1 if local_part != "" else 0
		feature_vector[4][0] = 1 if domain_part != "" else 0
		feature_vector[5][0] = 1 if '.' in domain_part else 0
		if local_part != "":
			feature_vector[6][0] = 1 if not local_part.startswith('.') and not local_part.endswith('.') else 0

	feature_vector[7][0] = 1 if maybe_email.endswith('.com') else 0
	feature_vector[8][0] = 1 if maybe_email.endswith('.edu') else 0
	feature_vector[9][0] = 1 if maybe_email.endswith('.tw') else 0

	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	email_list = []
	with open(DATA_FILE, 'r') as f:
		for line in f:
			email_list.append(line.strip())
	return email_list


if __name__ == '__main__':
	main()
