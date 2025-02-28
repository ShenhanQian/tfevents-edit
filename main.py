import sys
from pathlib import Path
import glob
import tensorflow as tf

assert len(sys.argv) == 2
path = Path(sys.argv[1])
assert path.exists()
event_files = glob.glob(str(path / 'events.out.tfevents.*.*.*.?'))
print(event_files)

for input_event_file in event_files:
  print(input_event_file)
  
  writer = tf.summary.create_file_writer(str(path))
  with writer.as_default():
    for event in tf.compat.v1.train.summary_iterator(input_event_file):
      step = event.step
      for v in event.summary.value:
        print(v.tag)
        
        # edit tags
        tf.summary.scalar(v.tag.replace('Habitat', ' Habitat'), v.simple_value, step=step)
        
        writer.flush()
