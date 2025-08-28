class: CommandLineTool
cwlVersion: v1.2
id: wsclean
label: WSClean
doc: Runs WSClean on the input data to produce an image.

baseCommand: wsclean
arguments: [-verbose, -log-time, -no-update-model-required]

inputs:
  - id: msin
    type:
      - Directory
      - Directory[]
    inputBinding:
      position: 2
      shellQuote: false
      itemSeparator: ' '
  - id: tempdir
    type: string
    default: '.'
    inputBinding:
      position: 1
      shellQuote: false
      itemSeparator: ' '
      prefix: '-temp-dir'
  - id: cores
    type: int?
    default: 24
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-j'
  - id: size
    type: int[]?
    default: [22500, 22500]
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-size'
  - id: baseline_averaging
    type: float?
    default: 0.0
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-baseline-averaging'
  - id: minuv-l
    type: float?
    default: 80.0
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-minuv-l'
  - id: weight
    type:
      - string?
    default: briggs -1.5
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-weight'
  - id: parallel-reordering
    type: int?
    default: 6
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-parallel-reordering'
  - id: mgain
    type: float?
    default: 0.7
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-mgain'
  - id: data-column
    type: string?
    default: DATA
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-data-column'
  - id: auto-mask
    type: float?
    default: 3.0
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-auto-mask'
  - id: auto-threshold
    type: float?
    default: 1.0
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-auto-threshold'
  - id: pol
    type: string?
    default: i
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-pol'
  - id: name
    type: string?
    default: "image"
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-name'
  - id: scale
    type: string?
    default: "0.4asec"
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-scale'
  - id: taper-gaussian
    type: string?
    default: 1.2asec
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-taper-gaussian'
  - id: niter
    type: int
    default: 150000
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-niter'
  - id: multiscale-scale-bias
    type: float?
    default: 0.6
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-multiscale-scale-bias'
  - id: parallel-deconvolution
    type: int?
    default: 2600
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-parallel-deconvolution'
  - id: parallel-gridding
    type: int?
    default: 4
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-parallel-gridding'
  - id: multiscale
    type: boolean?
    default: true
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-multiscale'
  - id: multiscale-max-scales
    type: int?
    default: 9
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-multiscale-max-scales'
  - id: nmiter
    type: int?
    default: 9
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-nmiter'
  - id: channels-out
    type: int?
    default: 6
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-channels-out'
  - id: join-channels
    type: boolean?
    default: true
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-join-channels'
  - id: fit-spectral-pol
    type: int?
    default: 3
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-fit-spectral-pol'
  - id: deconvolution-channels
    type: int?
    default: 3
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-deconvolution-channels'
  - id: gridder
    type: string?
    default: wgridder
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-gridder'
  - id: apply-primary-beam
    type: boolean?
    default: true
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-apply-primary-beam'
  - id: use-differential-lofar-beam
    type: boolean?
    default: true
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-use-differential-lofar-beam'
  - id: facet-regions
    type: File?
    inputBinding:
      position: 1
      shellQuote: false
      prefix: '-facet-regions'

  - id: facet-options
    type:
      type: record
      name: facet_options
      fields:
        - name: facet-solutions
          type: File?
          inputBinding:
            prefix: '-apply-facet-solutions'
        - name: soltabs
          type: string[]?
          inputBinding:
            itemSeparator: ','
    default:
      facet-solutions: null
      soltabs: null

outputs:
  - id: MFS_image_pb
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-image-pb.fits'
  - id: MFS_image
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-image.fits'
  - id: MFS_residual_pb
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-residual-pb.fits'
  - id: MFS_residual
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-residual.fits'
  - id: MFS_model_pb
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-model-pb.fits'
  - id: MFS_model
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-model.fits'
  - id: MFS_psf
    type: File
    doc: The final primary beam corrected image.
    outputBinding:
      glob: '$(inputs.name)-MFS-psf.fits'
  - id: channel_model_images
    type: File[]
    doc: Per-channel model images required for the facet subtraction.
    outputBinding:
      glob: '$(inputs.name)-????-model*.fits'

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

requirements:
  - class: ShellCommandRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
  - class: ResourceRequirement
    coresMin: $(inputs.cores)

stdout: wsclean.log
stderr: wsclean_err.log
