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
    if (annotations.length == 0) {
      //      setAnnotations([value[value.length - 1]])
      value[0].tag = 'E1'
      console.log(value)
      setAnnotations(value)
    }
    else if (annotations.length == 1) {
      let first = value[0]
      let latest = value[value.length - 1]
      let middle = { start: 0, end: 0, color: '#FFA500', tag: 'REL' }
      latest.tag = 'E2'
      if (first.end > latest.start) {
        first.tag = 'E2'
        latest.tag = 'E1'
        middle.start = latest.end
        middle.end = first.start
        setAnnotations([latest, middle, first])
      } else {
        middle.start = first.end
        middle.end = latest.start
        setAnnotations([first, middle, latest])
      }
    }
    // this requires the value to be an array
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
