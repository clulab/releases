'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _react = require('react');

var _react2 = _interopRequireDefault(_react);

var _Icon = require('./Icon.js');

var _Icon2 = _interopRequireDefault(_Icon);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

// `Toolbar` displays the Euclid tool buttons.
var ParseError = function (_React$Component) {
  _inherits(ParseError, _React$Component);

  function ParseError() {
    _classCallCheck(this, ParseError);

    return _possibleConstructorReturn(this, (ParseError.__proto__ || Object.getPrototypeOf(ParseError)).apply(this, arguments));
  }

  _createClass(ParseError, [{
    key: 'render',
    value: function render() {
      return _react2.default.createElement(
        'div',
        { className: 'main-stage__error-container' },
        _react2.default.createElement(
          'div',
          { className: 'parse-error' },
          _react2.default.createElement(_Icon2.default, { symbol: 'error', wrapperClass: 'parse-error__icon' }),
          _react2.default.createElement(
            'h1',
            { className: 'parse-error__primary' },
            _react2.default.createElement(
              'span',
              null,
              'Parsing error'
            )
          ),
          _react2.default.createElement(
            'p',
            { className: 'parse-error__secondary' },
            _react2.default.createElement(
              'strong',
              null,
              'No parse trees were returned in the JSON.'
            )
          ),
          _react2.default.createElement(
            'p',
            { className: 'parse-error__tertiary' },
            _react2.default.createElement(
              'span',
              null,
              'Press space bar to enter a new query.'
            )
          )
        )
      );
    }
  }]);

  return ParseError;
}(_react2.default.Component);

exports.default = ParseError;