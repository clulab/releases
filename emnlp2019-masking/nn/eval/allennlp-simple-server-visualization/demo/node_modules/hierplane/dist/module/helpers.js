'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.isSingleSegment = isSingleSegment;
exports.getCollapsibleNodeIds = getCollapsibleNodeIds;
exports.colorToString = colorToString;
exports.assignNodeIds = assignNodeIds;
exports.findAllNodeTypes = findAllNodeTypes;
exports.generateStylesForNodeTypes = generateStylesForNodeTypes;
exports.translateSpans = translateSpans;

var _immutable = require('immutable');

var _immutable2 = _interopRequireDefault(_immutable);

var _merge = require('merge');

var _merge2 = _interopRequireDefault(_merge);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

/**
 * Whether a parse tree is a single segment or not. This function is called at the top of the tree
 * and is passed down to all child nodes.
 *
 * @param   {string}  kind The type of root node.
 * @return  {boolean}
 */
function isSingleSegment(kind) {
  return kind !== "top-level-and" && kind !== "and" ? true : false;
}

/**
 * A recursive function for concatenating the ids of nodes that are collapsible in a depth
 * first manner.
 *
 * Setting children to [] indicates the base case where map is acting on an empty array, therefore
 * not recursing. A set with that leaf node's id is returned in this case.
 */
function getCollapsibleNodeIds(_ref, singleSegment) {
  var _Immutable$Set;

  var id = _ref.id,
      _ref$children = _ref.children,
      children = _ref$children === undefined ? [] : _ref$children,
      kind = _ref.kind;

  /*
    We only want to capture the ids of nodes that are collapsible, and, therefore, only nodes that
    a) have children and b) are not "root" nodes (as root nodes are not collapsible).
    A root node depends on whether a parse tree is a comprised of one or many "segments".
    If it has many segments, then nodes at both the '0', i.e., ids of length 1, and '0.x', i.e.,
    ids of length 3, levels are roots. Otherwise, it is a single segment, and just the '0' level
    is the root.
  */

  var hasChildren = children.length > 0;
  var isRoot = id.length === 1;
  var isEventRoot = !singleSegment && id.length === 3 || singleSegment && isRoot;

  var dataCollapsible = hasChildren && !isRoot && !isEventRoot;
  var nodeId = dataCollapsible ? [id] : [];

  return hasChildren ? (_Immutable$Set = _immutable2.default.Set(nodeId)).union.apply(_Immutable$Set, _toConsumableArray(children.map(function (child) {
    return getCollapsibleNodeIds(child, singleSegment);
  }))) : _immutable2.default.Set();
}

// Filter color style out of the style object and return the value:
function colorToString() {
  var arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];

  return arr.filter(function (item) {
    return item.indexOf("color") === 0;
  }, "");
}

/**
 * Returns a copy of the node, where the node and all of it's descendants are assigned unique
 * identifiers. Uniqueness is only guaranteed within the scope of the provided tree.
 *
 * @param  {Node}   node
 * @param  {String} [prefix='']   A prefix to append to generated identifiers.
 * @param  {Number} [childIdx=0]  The index of the node in it's parent.
 * @return {Node}
 */
function assignNodeIds(node) {
  var prefix = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : '';
  var childIdx = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;

  var nodeCopy = _merge2.default.recursive(true, node);
  var isLeaf = !Array.isArray(nodeCopy.children) || nodeCopy.children.length === 0;
  if (!nodeCopy.id) {
    nodeCopy.id = '' + prefix + childIdx;
  }
  if (Array.isArray(nodeCopy.children)) {
    nodeCopy.children = nodeCopy.children.slice().map(function (node, idx) {
      return assignNodeIds(node, nodeCopy.id + '.', idx);
    });
  }
  return nodeCopy;
}

/**
 * Returns an Immutable.Set including all unique nodeTypes discovered in the tree and all of it's
 * descendants.
 *
 * @param  {Node}                   node
 * @return {Immutable.Set<String>}  All unique nodeType values present in the tree.
 */
function findAllNodeTypes(node) {
  var nodeTypes = _immutable2.default.Set([node.nodeType]);
  if (Array.isArray(node.children)) {
    return node.children.reduce(function (types, node) {
      return types.concat(findAllNodeTypes(node));
    }, nodeTypes);
  } else {
    return nodeTypes;
  }
}

/**
 * Generates a map of node types to styles, for the provided node types.
 *
 * @param  {Immutable.Set<String>}  nodeTypes The set of all node types for which styles should be defined.
 * @return {object}                 A dictionary where each key is a nodeType and each value is a collection
 *                                  of styles to be applied to that node.
 */
function generateStylesForNodeTypes(nodeTypes) {
  if (!(nodeTypes instanceof _immutable2.default.Set)) {
    throw new Error('You must provide an Immutable.Set of node types.');
  }
  return nodeTypes.reduce(function (nodeTypeToStyle, nodeType) {
    // We have colors 0 through 6.  Dyanmically assign them.
    return nodeTypeToStyle.set(nodeType, ['color' + (nodeTypeToStyle.size % 6 + 1)]);
  }, _immutable2.default.Map()).toJS();
}

/**
 * Returns a copy fo the node and all of it's descendants, translating the generic `spans` interface
 * into `alternateParseInfo` as appropriate. This method was written to support translation from
 * a "public", easy to digest API into that which the existing UI / API expects.
 *
 * TODO (codeviking): In the long run we should remove this mechanism and use a more canonical API.
 *
 * @param  {Node}   origNode
 * @return {Node}   node  The same node, mutated.
 */
function translateSpans(origNode) {
  var node = _merge2.default.recursive(true, origNode);

  // First translate all of this node's children
  if (Array.isArray(node.children)) {
    node.children = node.children.map(translateSpans);
  }

  // If the property already exists, we assume it's data being delivered by Euclid's API, in which
  // case we shouldn't mutate the tree.
  if (!node.alternateParseInfo) {
    // First we build up alternateParseInfo.charNodeRoot, which is a single span that captures the
    // aggregate boundaries of the span and all of it's children.
    var boundaries = getSpanBoundaries(node);
    var charNodeRoot = boundaries ? new CharNodeRoot(boundaries.start, boundaries.end) : undefined;

    // TODO (codeviking): The UI should really support it being `undefined`, rather that using
    // if node.hasOwnProperty('charNodeRoot'), as then we wouldn't have to have carefully
    // implemented logic like so.
    if (charNodeRoot) {
      node.alternateParseInfo = { charNodeRoot: charNodeRoot };
    }

    // Now let's build up spanAnnotations, which are the aggregate boundaries (charNodeRoot) of the
    // node's immediate children and the node's own spans.
    var spanAnnotations = (node.children || []).filter(function (n) {
      return n.alternateParseInfo && n.alternateParseInfo.charNodeRoot;
    }).map(function (n) {
      return new Span(
      /* lo = */n.alternateParseInfo.charNodeRoot.charLo,
      /* hi = */n.alternateParseInfo.charNodeRoot.charHi,
      /* spanType = */'child');
    }).concat((node.spans || []).map(function (span) {
      return new Span(
      /* lo = */span.start,
      /* hi = */span.end,
      /* spanType = */span.spanType || 'self');
    })).sort(function (first, second) {
      return first.lo - second.lo;
    });

    // TODO (codeviking): Again, the UI should handle the "empty state" appropriately as to prevent
    // logic like this from being necessary.
    if (spanAnnotations.length > 0) {
      if (!node.alternateParseInfo) {
        node.alternateParseInfo = {};
      }
      node.alternateParseInfo.spanAnnotations = spanAnnotations;
    }
  }

  return node;
}

/**
 * Returns a single span where the the start / end values encompass the indices of the provided
 * node's spans and all of it's children's spans.
 *
 * For instance, if provided a node with the span [0, 1] and that node had two children,
 * [1, 3] and [4, 20], this function would return a single span, [0, 20].
 *
 * If the node or it's children don't have any spans, `undefined` is returned.
 *
 * @param  {Node}
 * @return {Span|undefined} The encompassing span (the boundaries), or undefined.
 */
function getSpanBoundaries(node) {
  var allSpans = getAllChildSpans(node).concat(node.spans || []);
  if (allSpans.length > 0) {
    var firstSpan = allSpans[0];
    return allSpans.reduce(function (boundaries, span) {
      if (boundaries.start > span.start) {
        boundaries.start = span.start;
      }
      if (boundaries.end < span.end) {
        boundaries.end = span.end;
      }
      return boundaries;
    }, { start: firstSpan.start, end: firstSpan.end });
  } else {
    return undefined;
  }
}

/**
 * Returns all children of the provided node, including those that are descendents of the node's
 * children.
 *
 * @param  {Node}   node
 * @return {Span[]}
 */
function getAllChildSpans(node) {
  return Array.isArray(node.children) ? node.children.map(function (n) {
    return (n.spans || []).concat(getAllChildSpans(n));
  }).reduce(function (all, arr) {
    return all.concat(arr);
  }) : [];
}

var Span = function Span(lo, hi, spanType) {
  _classCallCheck(this, Span);

  this.lo = lo;
  this.hi = hi;
  this.spanType = spanType;
};

var CharNodeRoot = function CharNodeRoot(charLo, charHi) {
  _classCallCheck(this, CharNodeRoot);

  this.charLo = charLo;
  this.charHi = charHi;
};