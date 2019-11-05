declare class Collapsible extends React.Component<CollapsibleProps> {}
declare namespace Collapsible {}

interface CollapsibleProps extends React.HTMLProps<Collapsible> {
  transitionTime?: number
  transitionCloseTime?: number | null
  triggerTagName?: string
  easing?: string
  open?: boolean
  classParentString?: string
  openedClassName?: string
  triggerStyle?: null | Object
  triggerClassName?: string
  triggerOpenedClassName?: string
  contentOuterClassName?: string
  contentInnerClassName?: string
  accordionPosition?: string | number
  handleTriggerClick?: (accordionPosition?: string | number) => void
  onOpen?: () => void
  onClose?: () => void
  onOpening?: () => void
  onClosing?: () => void
  trigger: string | React.ReactElement<any>
  triggerWhenOpen?: string | React.ReactElement<any>
  triggerDisabled?: boolean
  lazyRender?: boolean
  overflowWhenOpen?:
    | 'hidden'
    | 'visible'
    | 'auto'
    | 'scroll'
    | 'inherit'
    | 'initial'
    | 'unset'
  triggerSibling?: React.ReactElement<any>
  className?: string
  tabIndex?: number
}

declare module 'react-collapsible' {
  export default Collapsible
}
