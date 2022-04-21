import React, { useState } from 'react';
import { Box, Button, Typography } from '@mui/material';
import { TokenAnnotator } from 'react-text-annotate'

export function Annotator(props) {
  //
  const [annotations, setAnnotations] = useState([])
  // when you click annotate, it will set the editMode to true
  // whenever you finish editing stuff you set editMode to false
  const [editMode, setEditMode] = useState(false)
  const [showAnnotationButton, setShowAnnotationButton] = useState(false)
  //
  const handleChange = (value) => {
    // this requires the value to be an array
    setAnnotations([value[value.length - 1]])
  }
  //
  const handleSaveClick = () => {
    // send annotations with a callback
    // TODO: this also needs to receive:
    // TODO: docId
    // TODO: sentId
    props.onClickAnnotation(
      {
        'text': props.text,
        // here we return only one annotation
        'span': annotations[annotations.length - 1],
        'docId': props.docId,
        'sentId': props.sentId
      }
    )
    // clean state
    setShowAnnotationButton(false)
    setEditMode(false)
    setAnnotations([])
    //
  }
  //
  const toggleEdit = () => {
    setEditMode(!editMode)
  }

  //
  return (
    <Box>
      {editMode && (
        <Box>
          <Button variant='contained' color='secondary' onClick={() => setAnnotations([])}>
            Clear
          </Button>
          <Button variant='contained' color='primary' onClick={handleSaveClick}>
            Save
          </Button>
          <Button variant='contained' onClick={toggleEdit}>
            Cancel
          </Button>
          <Box p={2}>
            <TokenAnnotator
              tokens={props.text.split(' ')}
              value={annotations}
              getSpan={span => ({
                ...span,
                color: '#EEE',
                tag: 'ann',
              })}
              onChange={handleChange}
            />
          </Box>
        </Box>
      )}
      {!editMode && (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          onMouseEnter={() => setShowAnnotationButton(true)}
          onMouseLeave={() => setShowAnnotationButton(false)}
        >
          <Box flexGrow={1} o={2}>
            <Typography variant='body1' >
              {props.text}
            </Typography>
          </Box>
          {showAnnotationButton && (
            <Box p={2}>
              <Button variant='contained' onClick={toggleEdit}>
                Annotate
              </Button>
            </Box>
          )}
        </Box>
      )}
    </Box>
  )


}

export function AnnotatorContext(props) {
  const [annotations, setAnnotations] = useState([])
  // when you click annotate, it will set the editMode to true
  // whenever you finish editing stuff you set editMode to false
  const [editMode, setEditMode] = useState(false)
  const [showAnnotationButton, setShowAnnotationButton] = useState(false)

  //
  const handleChange = (value) => {
    // this requires the value to be an array
    if (annotations.length == 0) {
      let last = value[value.length -1]
      last.tag = 'context'
      setAnnotations([last])
    } else {
      let first = annotations[0]
      let last = value[value.length -1]

      last.tag = 'entity'
      last.color = '#FFA500'
      // TODO: enforce adjacency
      setAnnotations([first,last])
    }
  }

  //
  const handleSaveClick = () => {
    // TODO: split mask and selection
    // send annotations with a callback
    // add mask
    let mask = annotations[1]
    //
    props.onClickAnnotation(
      {
        'text': props.text,
        // here we return only one annotation
        'span': annotations[0],
        //"captures": [{"span": {"start": 4, "end": 5 } }],
        'captures': [{'span':{'start': mask.start, 'end': mask.end}}],
        'docId': props.docId,
        'sentId': props.sentId
      }
    )
    // clean state
    setShowAnnotationButton(false)
    setEditMode(false)
    setAnnotations([])
    //
  }

  //
  const toggleEdit = () => {
    setEditMode(!editMode)
  }
  //
  return (
    <Box>
      {editMode && (
        <Box>
          <Button variant='contained' color='secondary' onClick={() => setAnnotations([])}>
            Clear
          </Button>
          <Button variant='contained' color='primary' onClick={handleSaveClick}>
            Save
          </Button>
          <Button variant='contained' onClick={toggleEdit}>
            Cancel
          </Button>
          <Box p={2}>
            <TokenAnnotator
              tokens={props.text.split(' ')}
              value={annotations}
              getSpan={span => ({
                ...span,
                color: '#EEE',
                tag: 'ann',
              })}
              onChange={handleChange}
            />
          </Box>
        </Box>
      )}
      {!editMode && (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          onMouseEnter={() => setShowAnnotationButton(true)}
          onMouseLeave={() => setShowAnnotationButton(false)}
        >
          <Box flexGrow={1} o={2}>
            <Typography variant='body1' >
              {props.text}
            </Typography>
          </Box>
          {showAnnotationButton && (
            <Box p={2}>
              <Button variant='contained' onClick={toggleEdit}>
                Annotate
              </Button>
            </Box>
          )}
        </Box>
      )}
    </Box>
  )


}
