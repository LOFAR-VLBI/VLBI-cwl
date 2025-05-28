cwlVersion: v1.2
class: CommandLineTool
label: Get fraction of memory
doc: |
    Queries the available amount of memory on the system,
    and computes the given fraction of it in mebibytes (MiB).
    This is used to ensure that flagging jobs don't exceed the
    node's resources.

baseCommand: query_memory.py

stdout: memory.txt

inputs:
    - id: fraction
      type: int 
      inputBinding:
        position: 1
      doc: |
        The required fraction of the node's
        available memory for a flagging job.

outputs:
    - id: memory
      type: int 
      doc: |
        The fraction of the node's memory in
        mebibytes (MiB).
      outputBinding:
        glob: memory.json
        loadContents: true
        outputEval: $(JSON.parse(self[0].contents).memory)

hints:
    - class: DockerRequirement
      dockerPull: vlbi-cwl

requirements:
    - class: InlineJavascriptRequirement

