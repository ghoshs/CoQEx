import subprocess
from subprocess import PIPE

def run_subprocess(args):
	# completed_task = subprocess.run(args, capture_output=True)
	##server edit 
	completed_task = subprocess.run(args, stdout=PIPE, stderr=PIPE)
	result = completed_task.stdout
	if result is not None:
		return result.decode('utf-8')
	else:
		return ''