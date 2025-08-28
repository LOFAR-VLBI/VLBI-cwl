class: CommandLineTool
cwlVersion: v1.2
id: cut_fits_with_region
doc: Cut the given FITS image to the size of the provided DS9 region.

baseCommand: cut_fits_with_region

inputs:
  - id: region
    type: File
    doc: DS9 region file describing the area the image will be trimmed to.
    inputBinding:
      position: 0
      prefix: '--region'
  - id: output_name
    type: string
    doc: Name for the output file.
    inputBinding:
      position: 0
      prefix: '--output_name'
  - id: image
    type: File
    doc: FITS image to trim with the given region.
    inputBinding:
      position: 1

outputs:
  - id: trimmed_image
    type: File
    doc: FITS image trimmed to the provided region.
    outputBinding:
      glob: "$(inputs.output_name)"

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

