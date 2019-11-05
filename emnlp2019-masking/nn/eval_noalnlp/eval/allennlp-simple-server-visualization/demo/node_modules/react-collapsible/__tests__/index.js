// Tests for Collapsible.
import React from 'react';
import { configure, shallow, mount, render } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

import Collapsible from '../src/Collapsible';

configure({ adapter: new Adapter() });

const dummyEvent = { preventDefault: () => {}};

describe('<Collapsbile />', () => {
  it('renders an element with the class `.Collapsible`.', () => {
    const wrapper = shallow(<Collapsible />);
    expect(wrapper.is('.Collapsible')).toEqual(true);
  });

  it('renders Collapsible with trigger text.', () => {
    const wrapper = shallow(<Collapsible trigger='Hello World'/> )
    expect(wrapper.find('span').text()).toEqual('Hello World')
  });

  it('given a closed Collapsible fires the onOpening prop when clicked to open', () => {
    const mockOnOpening = jest.fn();
    const collapsbile = shallow(<Collapsible trigger='Hello World' onOpening={mockOnOpening}/> );
    const trigger = collapsbile.find('.Collapsible__trigger');

    expect(trigger).toHaveLength(1);
    trigger.simulate('click', dummyEvent);
    expect(mockOnOpening.mock.calls).toHaveLength(1);
  });

  it('given an open Collapsible fires the onClosing prop when clicked to close', () => {
    const mockOnClosing = jest.fn();
    const collapsbile = mount(<Collapsible open trigger='Hello World' onClosing={mockOnClosing}/> );
    const trigger = collapsbile.find('.Collapsible__trigger');

    expect(trigger).toHaveLength(1);
    trigger.simulate('click', dummyEvent);
    expect(mockOnClosing.mock.calls).toHaveLength(1);
  });

  it('given a closed Collapsible it fires the onOpen prop after the transistion', () => {
    const mockOnOpen = jest.fn();
    const collapsbile = shallow(<Collapsible open trigger='Hello World' onOpen={mockOnOpen}>Some Content</Collapsible> );
    const outer = collapsbile.find('.Collapsible__contentOuter');
    
    expect(outer).toHaveLength(1);
    outer.simulate('transitionEnd', dummyEvent);
    expect(mockOnOpen.mock.calls).toHaveLength(1);
  });

  it('given an open Collapsible it fires the onClose prop after the transistion', () => {
    const mockOnClose = jest.fn();
    const collapsbile = shallow(<Collapsible trigger='Hello World' onClose={mockOnClose}>Some Content</Collapsible> );
    const outer = collapsbile.find('.Collapsible__contentOuter');
    
    expect(outer).toHaveLength(1);
    outer.simulate('transitionEnd', dummyEvent);
    expect(mockOnClose.mock.calls).toHaveLength(1);
  });

  it('given a Collapsible with the handleTriggerClick prop, the handleTriggerClick prop gets fired', () => {
    const mockHandleTriggerClick = jest.fn();
    const collapsbile = shallow(<Collapsible handleTriggerClick={mockHandleTriggerClick} trigger="Hello world" />);
    const trigger = collapsbile.find('.Collapsible__trigger');

    expect(trigger).toHaveLength(1);
    trigger.simulate('click', dummyEvent);
    expect(mockHandleTriggerClick.mock.calls).toHaveLength(1);
  })
})