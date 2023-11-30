// ignore_for_file: prefer_const_constructors

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class SummarizedText extends StatelessWidget {
  const SummarizedText({Key? key , required this.sumedText}) : super(key: key);

  final String sumedText;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: Text('Summary'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(5.0),
        child: Container(
          color: Colors.white,
          child: Text(
            sumedText,
            //'not as advertised',
            //'delicious',
            style:TextStyle(
                      fontSize: 23,
                      fontWeight: FontWeight.bold,
                      
                    )
          ),
        ),
      ),
    );
  }
}
