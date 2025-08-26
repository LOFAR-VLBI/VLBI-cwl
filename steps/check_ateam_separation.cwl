class: CommandLineTool
cwlVersion: v1.2
id: check_ateam_separation
label: check Ateam separation
doc: |
    Checks whether the source is sufficiently
    far away from A-team sources. Produces an
    image of the plotted distance.

baseCommand: check_Ateam_separation.py

inputs:
  - id: ms
    type:
      - Directory
      - type: array
        items: Directory
    inputBinding:
      position: 0
    doc: An array of input data in MeasurementSet format.

  - id: output_image_name
    default: Ateam_separation.png
    type: string?
    inputBinding:
      position: 2
      prefix: --outputimage
    doc: The filename of the output image.

  - id: min_separation
    type: int?
    inputBinding:
      position: 1
      prefix: --min_separation
    doc: |
        The minimal accepted distance to an
        A-team source on the sky in degrees.

outputs:
  - id: output_image
    doc: The output image containing the plotted distances.
    type: File
    outputBinding:
      glob: $(inputs.output_image_name)

  - id: output_json
    doc: Distances to A-team sources in JSON format.
    type: File
    outputBinding:
      glob: '*.json'

  - id: logfile
    type: File
    outputBinding:
      glob: Ateam_separation.log
    doc: |
        The files containing the stdout
        and stderr from the step.

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: InlineJavascriptRequirement

stdout: Ateam_separation.log
