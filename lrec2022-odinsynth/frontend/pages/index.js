import React from 'react';
import { Container, Box, Button, CssBaseline } from '@mui/material';
import Link from 'next/link'


 export default function HomeApp() {
  return (
    <Box>
      <CssBaseline />
      <Container maxWidth='sm' >
        <Box>
          <Link
            href={{ pathname: '/entity' }}
          >
            <Button variant='contained'>Entities</Button>
          </Link>
        </Box>
        <Box>
          <Link
            href={{ pathname: '/context' }}
          >
            <Button variant='contained'>Context</Button>
          </Link>
        </Box>
      </Container>
    </Box>
  );
}
