// ignore_for_file: prefer_const_constructors, unused_element, unused_field

import 'package:flutter/material.dart';
import 'package:sumit/text_page.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';

void main() {
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Summarizer Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const TextPage(),
      debugShowCheckedModeBanner: false,
      builder: EasyLoading.init(),
    );
  }
}
