'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.renderTree = renderTree;

var _index = require('../module/index.js');

var _react = require('react');

var _react2 = _interopRequireDefault(_react);

var _reactDom = require('react-dom');

var _reactDom2 = _interopRequireDefault(_reactDom);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Renders a hierplane tree visualization from the provided tree.
 *
 * @param  {Object} tree                      The tree to render.
 * @param  {Object} [options]                 Optional command options.
 * @param  {string} [options.target='body']   The element into which the tree should be rendered, this
 *                                            defaults to document.body.
 * @param  {string} [options.theme=undefined] The theme to use, can be "light" or undefined.
 * @return {undefined}
 */
function renderTree(tree) {
  var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : { target: 'body' };

  _reactDom2.default.render(_react2.default.createElement(_index.Tree, { tree: tree, theme: options.theme ? options.theme : undefined }), document.querySelector(options.target));
}