import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() {
  runApp(const StockPredictorApp());
}

class StockPredictorApp extends StatelessWidget {
  const StockPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'S&P 500 Close Price Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0A2540),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF0A1628),
        fontFamily: 'monospace',
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage>
    with SingleTickerProviderStateMixin {
  // ── Controllers ──────────────────────────────────────────
  final _openCtrl         = TextEditingController();
  final _highCtrl         = TextEditingController();
  final _lowCtrl          = TextEditingController();
  final _volumeCtrl       = TextEditingController();
  final _rollingAvgCtrl   = TextEditingController();
  final _formKey           = GlobalKey<FormState>();

  // ── State ─────────────────────────────────────────────────
  bool   _isLoading       = false;
  String? _resultPrice;
  String? _errorMessage;
  late AnimationController _animCtrl;
  late Animation<double>   _fadeAnim;

  static const String _apiUrl =
      'https://stock-price-predictor-cetx.onrender.com/predict';

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _openCtrl.dispose();
    _highCtrl.dispose();
    _lowCtrl.dispose();
    _volumeCtrl.dispose();
    _rollingAvgCtrl.dispose();
    _animCtrl.dispose();
    super.dispose();
  }

  // ── API Call ──────────────────────────────────────────────
  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading    = true;
      _resultPrice  = null;
      _errorMessage = null;
    });
    _animCtrl.reset();

    try {
      final body = jsonEncode({
        'open'          : double.parse(_openCtrl.text.trim()),
        'high'          : double.parse(_highCtrl.text.trim()),
        'low'           : double.parse(_lowCtrl.text.trim()),
        'volume'        : double.parse(_volumeCtrl.text.trim()),
        'rolling_avg_5d': double.parse(_rollingAvgCtrl.text.trim()),
      });

      final response = await http
          .post(
            Uri.parse(_apiUrl),
            headers: {
              'Content-Type': 'application/json',
              'Accept'      : 'application/json',
            },
            body: body,
          )
          .timeout(const Duration(seconds: 60));

      final data = jsonDecode(response.body);

      if (response.statusCode == 200) {
        setState(() {
          _resultPrice = '\$${data['predicted_close_price'].toStringAsFixed(2)}';
        });
      } else if (response.statusCode == 422) {
        // Pydantic validation error
        final detail = data['detail'];
        String msg = 'Validation error: ';
        if (detail is List && detail.isNotEmpty) {
          msg += detail[0]['msg'] ?? 'Invalid input values.';
        } else {
          msg += detail.toString();
        }
        setState(() => _errorMessage = msg);
      } else {
        setState(() => _errorMessage = 'Server error (${response.statusCode}). Please try again.');
      }
    } on FormatException {
      setState(() => _errorMessage = 'All fields must be valid numbers.');
    } catch (e) {
      setState(() => _errorMessage = 'Connection error. Check your internet and try again.\n(Note: Free API may take ~50s to wake up)');
    } finally {
      setState(() => _isLoading = false);
      _animCtrl.forward();
    }
  }

  void _clearAll() {
    _openCtrl.clear();
    _highCtrl.clear();
    _lowCtrl.clear();
    _volumeCtrl.clear();
    _rollingAvgCtrl.clear();
    setState(() {
      _resultPrice  = null;
      _errorMessage = null;
    });
    _animCtrl.reset();
  }

  // ── UI ────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),
                const SizedBox(height: 28),
                _buildSectionLabel('MARKET DATA INPUT'),
                const SizedBox(height: 14),
                _buildInputField(
                  controller  : _openCtrl,
                  label       : 'Open Price',
                  hint        : 'e.g. 145.30',
                  icon        : Icons.open_in_new_rounded,
                  prefix      : '\$',
                  helperText  : 'Stock price at market open (USD)',
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(
                      child: _buildInputField(
                        controller: _highCtrl,
                        label     : 'Daily High',
                        hint      : 'e.g. 148.20',
                        icon      : Icons.arrow_upward_rounded,
                        prefix    : '\$',
                        iconColor : const Color(0xFF00C853),
                        helperText: 'Highest price of the day',
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildInputField(
                        controller: _lowCtrl,
                        label     : 'Daily Low',
                        hint      : 'e.g. 144.10',
                        icon      : Icons.arrow_downward_rounded,
                        prefix    : '\$',
                        iconColor : const Color(0xFFFF5252),
                        helperText: 'Lowest price of the day',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                _buildInputField(
                  controller : _volumeCtrl,
                  label      : 'Trading Volume',
                  hint       : 'e.g. 12000000',
                  icon       : Icons.bar_chart_rounded,
                  prefix     : '#',
                  helperText : 'Total shares traded today',
                  isVolume   : true,
                ),
                const SizedBox(height: 14),
                _buildInputField(
                  controller : _rollingAvgCtrl,
                  label      : '5-Day Rolling Average',
                  hint       : 'e.g. 144.80',
                  icon       : Icons.show_chart_rounded,
                  prefix     : '\$',
                  helperText : 'Average closing price over past 5 days',
                ),
                const SizedBox(height: 28),
                _buildPredictButton(),
                const SizedBox(height: 20),
                _buildResultArea(),
                const SizedBox(height: 12),
                _buildDisclaimer(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color       : const Color(0xFF1565C0).withOpacity(0.3),
                borderRadius: BorderRadius.circular(12),
                border      : Border.all(color: const Color(0xFF1E88E5).withOpacity(0.5)),
              ),
              child: const Icon(Icons.candlestick_chart_rounded,
                  color: Color(0xFF42A5F5), size: 28),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'S&P 500 Predictor',
                    style: TextStyle(
                      color     : Colors.white,
                      fontSize  : 20,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                  Text(
                    'Closing Price Forecast',
                    style: TextStyle(
                      color   : Colors.white.withOpacity(0.5),
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            IconButton(
              onPressed: _clearAll,
              icon: const Icon(Icons.refresh_rounded, color: Color(0xFF42A5F5)),
              tooltip: 'Clear all fields',
            ),
          ],
        ),
        const SizedBox(height: 16),
        Container(
          height: 1,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                const Color(0xFF1E88E5).withOpacity(0.8),
                const Color(0xFF1E88E5).withOpacity(0.0),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSectionLabel(String label) {
    return Text(
      label,
      style: TextStyle(
        color        : const Color(0xFF42A5F5).withOpacity(0.8),
        fontSize     : 11,
        fontWeight   : FontWeight.bold,
        letterSpacing: 2.0,
      ),
    );
  }

  Widget _buildInputField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    required String prefix,
    required String helperText,
    Color? iconColor,
    bool isVolume = false,
  }) {
    return TextFormField(
      controller       : controller,
      keyboardType     : const TextInputType.numberWithOptions(decimal: true),
      inputFormatters  : [
        FilteringTextInputFormatter.allow(RegExp(r'[0-9.]')),
      ],
      style            : const TextStyle(color: Colors.white, fontSize: 15),
      decoration       : InputDecoration(
        labelText     : label,
        hintText      : hint,
        helperText    : helperText,
        helperMaxLines: 1,
        prefixIcon    : Icon(icon, color: iconColor ?? const Color(0xFF42A5F5), size: 20),
        prefixText    : '  $prefix ',
        prefixStyle   : TextStyle(
          color     : (iconColor ?? const Color(0xFF42A5F5)).withOpacity(0.7),
          fontSize  : 14,
          fontWeight: FontWeight.bold,
        ),
        labelStyle : TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 13),
        hintStyle  : TextStyle(color: Colors.white.withOpacity(0.25), fontSize: 13),
        helperStyle: TextStyle(color: Colors.white.withOpacity(0.35), fontSize: 11),
        filled      : true,
        fillColor   : const Color(0xFF0D2137),
        border      : OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide  : BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide  : BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide  : const BorderSide(color: Color(0xFF1E88E5), width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide  : const BorderSide(color: Color(0xFFFF5252)),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide  : const BorderSide(color: Color(0xFFFF5252), width: 1.5),
        ),
        errorStyle: const TextStyle(color: Color(0xFFFF5252), fontSize: 11),
      ),
      validator: (value) {
        if (value == null || value.trim().isEmpty) return '$label is required';
        final parsed = double.tryParse(value.trim());
        if (parsed == null) return 'Enter a valid number';
        if (parsed <= 0) return 'Must be greater than 0';
        if (!isVolume && parsed > 5000) return 'Must be ≤ 5000';
        return null;
      },
    );
  }

  Widget _buildPredictButton() {
    return SizedBox(
      width : double.infinity,
      height: 54,
      child : ElevatedButton(
        onPressed: _isLoading ? null : _predict,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF1565C0),
          foregroundColor: Colors.white,
          disabledBackgroundColor: const Color(0xFF1565C0).withOpacity(0.4),
          elevation   : 0,
          shape       : RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        child: _isLoading
            ? const SizedBox(
                width : 22,
                height: 22,
                child : CircularProgressIndicator(
                  color      : Colors.white,
                  strokeWidth: 2.5,
                ),
              )
            : const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.auto_graph_rounded, size: 20),
                  SizedBox(width: 10),
                  Text(
                    'Predict',
                    style: TextStyle(
                      fontSize  : 16,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.0,
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_resultPrice == null && _errorMessage == null) {
      return Container(
        width      : double.infinity,
        padding    : const EdgeInsets.all(20),
        decoration : BoxDecoration(
          color       : const Color(0xFF0D2137),
          borderRadius: BorderRadius.circular(14),
          border      : Border.all(color: Colors.white.withOpacity(0.07)),
        ),
        child: Column(
          children: [
            Icon(Icons.query_stats_rounded,
                color: Colors.white.withOpacity(0.2), size: 36),
            const SizedBox(height: 10),
            Text(
              'Prediction will appear here',
              style: TextStyle(
                color   : Colors.white.withOpacity(0.3),
                fontSize: 14,
              ),
            ),
          ],
        ),
      );
    }

    return FadeTransition(
      opacity: _fadeAnim,
      child: _resultPrice != null
          ? _buildSuccessResult()
          : _buildErrorResult(),
    );
  }

  Widget _buildSuccessResult() {
    return Container(
      width      : double.infinity,
      padding    : const EdgeInsets.all(24),
      decoration : BoxDecoration(
        color       : const Color(0xFF003300).withOpacity(0.4),
        borderRadius: BorderRadius.circular(14),
        border      : Border.all(color: const Color(0xFF00C853).withOpacity(0.4)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.check_circle_rounded,
                  color: Color(0xFF00C853), size: 18),
              const SizedBox(width: 8),
              Text(
                'PREDICTED CLOSING PRICE',
                style: TextStyle(
                  color        : const Color(0xFF00C853).withOpacity(0.8),
                  fontSize     : 11,
                  fontWeight   : FontWeight.bold,
                  letterSpacing: 1.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            _resultPrice!,
            style: const TextStyle(
              color     : Colors.white,
              fontSize  : 42,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.0,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Linear Regression Model · USD',
            style: TextStyle(
              color   : Colors.white.withOpacity(0.4),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorResult() {
    return Container(
      width      : double.infinity,
      padding    : const EdgeInsets.all(18),
      decoration : BoxDecoration(
        color       : const Color(0xFF330000).withOpacity(0.4),
        borderRadius: BorderRadius.circular(14),
        border      : Border.all(color: const Color(0xFFFF5252).withOpacity(0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline_rounded,
              color: Color(0xFFFF5252), size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _errorMessage!,
              style: const TextStyle(
                color   : Color(0xFFFF7070),
                fontSize: 13,
                height  : 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Center(
      child: Text(
        'For educational purposes only · Not financial advice',
        style: TextStyle(
          color   : Colors.white.withOpacity(0.2),
          fontSize: 10,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
