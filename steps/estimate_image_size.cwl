class: CommandLineTool
cwlVersion: v1.2
id: estimate_facet_size
doc: Estimates the appropriate image size and baseline averaging for imaging the facet.

baseCommand:
  - estimate_facet_size.py

inputs:
  - id: region
    type: File
    doc: DS9 region file describing the facet.
    inputBinding:
      position: 0
      prefix: '--region'
  - id: resolution
    type: string
    doc: Angular resolution that will be passed to WSClean. Used here for naming the output image.
    inputBinding:
      position: 0
      prefix: '--resolution'
  - id: pixel_size
    type: float
    doc: Pixel size in arcseconds of the image that will be made.
    inputBinding:
      position: 0
      prefix: '--pixel_size'
  - id: padding
    type: float?
    doc: Factor by which to multiply the calculated image size.
    default: 1.0
    inputBinding:
      position: 0
      prefix: '--padding'
  - id: filename
    type: string?
    doc: |
      Output filename to which the image parameters are written. Do not change this.
      cwl.output.json is a special name for CWL to directly return its contents via the
      output bindings.
    default: 'cwl.output.json'
    inputBinding:
      position: 0
      prefix: '--filename'

outputs:
  - id: name
    type: string
    doc: Name of the image that will be created.
  - id: image_size
    type: int[]
    doc: Width and height of the image in pixels.
  - id: blavg
    type: float
    doc: Baseline averaging factor corresponding to the image size.

requirements:
    - class: InlineJavascriptRequirement
